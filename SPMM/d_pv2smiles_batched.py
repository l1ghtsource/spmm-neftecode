import argparse
import torch
import os
import numpy as np
from SPMM_models import SPMM
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer, WordpieceTokenizer
from torch.utils.data import Dataset, DataLoader
from calc_property import calculate_property
from torch.distributions.categorical import Categorical
from rdkit import Chem
import random
import pandas as pd
import pickle
import warnings
from tqdm import tqdm
from bisect import bisect_left
warnings.filterwarnings(action='ignore')

class SMILESDatasetFromCSV(Dataset):
    def __init__(self, csv_file, property_positions=None):
        self.df = pd.read_csv(csv_file)
        if 'Smiles' not in self.df.columns:
            raise ValueError("CSV file must contain a 'Smiles' column")
        self.property_positions = property_positions if property_positions else []
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        smiles = self.df['Smiles'].iloc[idx]
        try:
            properties = calculate_property(smiles)
            
            prop_mask = torch.ones(53)
            for pos in self.property_positions:
                prop_mask[pos] = 0
            
            prop_input = torch.zeros(53)
            for pos in self.property_positions:
                prop_input[pos] = properties[pos]
                
            return prop_input, prop_mask, smiles
        except Exception as e:
            return torch.zeros(53), torch.ones(53), smiles

def BinarySearch(a, x):
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    else:
        return -1

def generate(model, image_embeds, text, stochastic=True, prop_att_mask=None, k=None):
    text_atts = torch.where(text == 0, 0, 1)
    if prop_att_mask is None:   
        prop_att_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
    token_output = model.text_encoder(text,
                                      attention_mask=text_atts,
                                      encoder_hidden_states=image_embeds,
                                      encoder_attention_mask=prop_att_mask,
                                      return_dict=True,
                                      is_decoder=True,
                                      return_logits=True,
                                      )[:, -1, :]  # batch*300
    if k:
        p = torch.softmax(token_output, dim=-1)
        if stochastic:
            output = torch.multinomial(p, num_samples=k, replacement=False)
            return torch.log(torch.stack([p[i][output[i]] for i in range(output.size(0))])), output
        else:
            output = torch.topk(p, k=k, dim=-1)  # batch*k
            return torch.log(output.values), output.indices
    if stochastic:
        p = torch.softmax(token_output, dim=-1)
        m = Categorical(p)
        token_output = m.sample()
    else:
        token_output = torch.argmax(token_output, dim=-1)
    return token_output.unsqueeze(1)  # batch*1

@torch.no_grad()
def generate_with_property_batch(model, data_loader, n_sample, tokenizer, device, stochastic=True, k=2):
    model.eval()
    print(f"PV-to-SMILES generation in {'stochastic' if stochastic else 'deterministic'} manner with k={k}...")

    with open('./normalize.pkl', 'rb') as w:
        norm = pickle.load(w)
    property_mean, property_std = norm
    
    all_results = []
    all_inputs = []
    all_masks = []
    all_smiles = []
    
    for batch_idx, (prop_inputs, prop_masks, smiles_list) in enumerate(tqdm(data_loader, desc="Processing batches")):
        valid_indices = [i for i, mask in enumerate(prop_masks) if not mask.all()]
        if not valid_indices:
            continue
            
        batch_props = torch.stack([prop_inputs[i] for i in valid_indices]).to(device)
        batch_masks = torch.stack([prop_masks[i] for i in valid_indices]).to(device)
        batch_smiles = [smiles_list[i] for i in valid_indices]
        
        batch_props = (batch_props - property_mean) / property_std
        
        batch_size = batch_props.size(0)
        batch_results = [[] for _ in range(batch_size)]
        
        property1 = model.property_embed(batch_props.unsqueeze(2))
        
        property_unk = model.property_mask.expand(batch_size, property1.size(1), -1)
        mpm_mask_expand = batch_masks.unsqueeze(2).expand(-1, -1, property_unk.size(2))
        property_masked = property1 * (1 - mpm_mask_expand) + property_unk * mpm_mask_expand
        
        properties = torch.cat([model.property_cls.expand(batch_size, -1, -1), property_masked], dim=1)
        prop_embeds = model.property_encoder(inputs_embeds=properties, return_dict=True).last_hidden_state
        
        for batch_item in range(batch_size):
            item_prop_embeds = prop_embeds[batch_item:batch_item+1]
            
            for _ in range(n_sample):
                product_input = torch.tensor([tokenizer.cls_token_id]).expand(1, 1).to(device)
                
                values, indices = generate(model, item_prop_embeds, product_input, stochastic=stochastic, k=k)
                product_input = torch.cat([
                    torch.tensor([tokenizer.cls_token_id]).expand(k, 1).to(device), 
                    indices.squeeze(0).unsqueeze(-1)
                ], dim=-1)
                current_p = values.squeeze(0)
                
                final_output = []
                for _ in range(100):
                    values, indices = generate(model, item_prop_embeds, product_input, stochastic=stochastic, k=k)
                    k2_p = current_p[:, None] + values
                    product_input_k2 = torch.cat([
                        product_input.unsqueeze(1).repeat(1, k, 1), 
                        indices.unsqueeze(-1)
                    ], dim=-1)
                    
                    if tokenizer.sep_token_id in indices:
                        ends = (indices == tokenizer.sep_token_id).nonzero(as_tuple=False)
                        for e in ends:
                            p = k2_p[e[0], e[1]].cpu().item()
                            final_output.append((p, product_input_k2[e[0], e[1]]))
                            k2_p[e[0], e[1]] = -1e5
                        if len(final_output) >= k**2:
                            break
                    
                    current_p, i = torch.topk(k2_p.flatten(), k)
                    next_indices = torch.from_numpy(np.array(np.unravel_index(i.cpu().numpy(), k2_p.shape))).T
                    product_input = torch.stack([
                        product_input_k2[i[0], i[1]] for i in next_indices
                    ], dim=0)
                
                if not final_output:
                    continue
                    
                final_output = sorted(final_output, key=lambda x: x[0], reverse=True)[:k]
                
                candidate_k = []
                for p, sentence in final_output:
                    cdd = tokenizer.convert_tokens_to_string(
                        tokenizer.convert_ids_to_tokens(sentence[:-1])
                    ).replace('[CLS]', '')
                    candidate_k.append(cdd)
                
                if not stochastic and candidate_k:
                    batch_results[batch_item].append(candidate_k[0])
                elif candidate_k:
                    batch_results[batch_item].append(random.choice(candidate_k))
        
        for i in range(batch_size):
            if batch_results[i]: 
                all_results.append(batch_results[i])
                all_inputs.append(batch_props[i].cpu())
                all_masks.append(batch_masks[i].cpu())
                all_smiles.append(batch_smiles[i])
    
    return all_results, all_inputs, all_masks, all_smiles

@torch.no_grad()
def metric_eval_batch(prop_inputs, candidates, masks):
    with open('./normalize.pkl', 'rb') as w:
        norm = pickle.load(w)
    property_mean, property_std = norm
    
    total_mse = []
    all_valids = []
    prop_cdds = []
    
    for i in range(len(candidates)):
        prop_input = prop_inputs[i]
        cand_list = candidates[i]
        mask = masks[i]
        
        random.shuffle(cand_list)
        mse_list = []
        valid_smiles = []
        
        for smiles in cand_list:
            try:
                prop_cdd = calculate_property(smiles)
                n_ref = (prop_input - property_mean) / property_std
                n_cdd = (torch.tensor(prop_cdd) - property_mean) / property_std
                mse_list.append((n_ref - n_cdd) ** 2)
                valid_smiles.append(smiles)
                prop_cdds.append(prop_cdd)
            except:
                continue
        
        if mse_list:
            mse = torch.stack(mse_list, dim=0)
            total_mse.append(mse)
            all_valids.extend(valid_smiles)
    
    if total_mse:
        mse = torch.cat(total_mse, dim=0)
        rmse = torch.sqrt(torch.mean(mse, dim=0))
        print('Mean of controlled properties\' normalized RMSE:', rmse[(1 - masks[0]).long().bool()].mean().item())
    else:
        print('No valid molecules generated for RMSE calculation')
    
    validity = len(all_valids) / sum(len(cand) for cand in candidates) if candidates else 0
    print('Validity:', validity)
    
    canonical_valids = [Chem.MolToSmiles(Chem.MolFromSmiles(l), isomericSmiles=False, canonical=True) for l in all_valids]
    unique_valids = list(set(canonical_valids))
    uniqueness = len(unique_valids) / len(all_valids) if all_valids else 0
    print('Uniqueness:', uniqueness)
    
    with open('generated_molecules.txt', 'w') as w:
        for v in all_valids:
            w.write(v + '\n')
    print('Generated molecules are saved in \'generated_molecules.txt\'')
    
    return all_valids

def main(args, config):
    device = torch.device(args.device)

    seed = random.randint(0, 1000) if not args.seed else args.seed
    print('seed:', seed, 'stochastic:', args.stochastic)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    tokenizer = BertTokenizer(vocab_file=args.vocab_filename, do_lower_case=False, do_basic_tokenize=False)
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token, max_input_chars_per_word=250)

    print("Creating model")
    model = SPMM(config=config, tokenizer=tokenizer, no_train=True)

    if args.checkpoint:
        print('LOADING PRETRAINED MODEL..')
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['state_dict']

        for key in list(state_dict.keys()):
            if 'word_embeddings' in key and 'property_encoder' in key:
                del state_dict[key]
            if 'queue' in key:
                del state_dict[key]

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)
    model = model.to(device)

    if not args.csv_file:
        print("Error: CSV file path is required. Use --csv_file argument.")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    dataset = SMILESDatasetFromCSV(args.csv_file, args.property_positions)
    data_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True, 
        drop_last=False
    )
    
    print("=" * 50)
    print(f"Processing {len(dataset)} molecules in batches of {args.batch_size}")
    
    all_results, all_inputs, all_masks, all_smiles = generate_with_property_batch(
        model, data_loader, args.n_generate, tokenizer, device, 
        stochastic=args.stochastic, k=args.k
    )
    
    all_valids = metric_eval_batch(all_inputs, all_results, all_masks)
    
    for i, (smiles, results) in enumerate(zip(all_smiles, all_results)):
        output_file = os.path.join(args.output_dir, f"generated_molecules_{i}.txt")
        with open(output_file, 'w') as f:
            f.write(f"Input SMILES: {smiles}\n")
            f.write("-" * 50 + "\n")
            for result in results:
                f.write(f"{result}\n")
        print(f"Saved {len(results)} generated molecules for input {i+1} to {output_file}")
    
    print("=" * 50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='./Pretrain/checkpoint_SPMM.ckpt')
    parser.add_argument('--vocab_filename', default='./vocab_bpe_300.txt')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--n_generate', default=10, type=int, help='Number of molecules to generate per input SMILES')
    parser.add_argument('--k', default=2, type=int, help='Number of candidates to consider at each step')
    parser.add_argument('--stochastic', default=False, type=bool, help='Whether to use stochastic or deterministic generation')
    parser.add_argument('--csv_file', required=True, help='Input CSV file with Smiles column')
    parser.add_argument('--output_dir', default='./generated_molecules', help='Directory to save generated molecules')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size for processing')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--property_positions', type=int, nargs='+', default=[30, 42], 
                        help='List of property positions to use (default: [30, 42] for MolLogP and NumHDonors)')
    args = parser.parse_args()

    configs = {
        'embed_dim': 256,
        'bert_config_text': './config_bert.json',
        'bert_config_property': './config_bert_property.json',
    }
    main(args, configs)
