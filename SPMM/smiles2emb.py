import argparse
import torch
import pandas as pd
import numpy as np
from SPMM_models import SPMM
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer, WordpieceTokenizer
import random

@torch.no_grad()
def get_smiles_embeddings(model, smiles_list, batch_size=64):
    """Extract embeddings for SMILES strings using only the text encoder"""
    device = model.device
    tokenizer = model.tokenizer
    model.eval()
    print("Generating SMILES embeddings...")
    
    all_embeddings = []
    
    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i:i+batch_size]
        
        processed_smiles = ['[CLS]'+s if not s.startswith("[CLS]") else s for s in batch_smiles]
        
        text_input = tokenizer(processed_smiles, padding='longest', truncation=True, 
                              max_length=100, return_tensors="pt").to(device)
        
        text_embeds = model.text_encoder.bert(
            text_input.input_ids[:, 1:], 
            attention_mask=text_input.attention_mask[:, 1:],
            return_dict=True, 
            mode='text'
        ).last_hidden_state
        
        mask = text_input.attention_mask[:, 1:].unsqueeze(-1)
        masked_embeddings = text_embeds * mask
        sum_embeddings = torch.sum(masked_embeddings, dim=1)
        sum_mask = torch.clamp(torch.sum(mask, dim=1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        
        batch_embeddings = mean_embeddings.cpu().numpy()
        all_embeddings.extend(batch_embeddings)
        
        if (i + batch_size) % (10 * batch_size) == 0 or (i + batch_size) >= len(smiles_list):
            print(f"Processed {min(i + batch_size, len(smiles_list))}/{len(smiles_list)} SMILES")
    
    return all_embeddings

def main(args, config):
    device = torch.device(args.device)

    seed = args.seed if args.seed is not None else random.randint(0, 1000)
    print('seed:', seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Loading CSV data")
    df = pd.read_csv(args.input_file)
    
    if args.smiles_column not in df.columns:
        raise ValueError(f"CSV must contain a '{args.smiles_column}' column")
    
    smiles_list = df[args.smiles_column].tolist()
    
    tokenizer = BertTokenizer(
        vocab_file=args.vocab_filename, 
        do_lower_case=False, 
        do_basic_tokenize=False
    )
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(
        vocab=tokenizer.vocab, 
        unk_token=tokenizer.unk_token, 
        max_input_chars_per_word=250
    )

    print("Creating model")
    model = SPMM(config=config, tokenizer=tokenizer, no_train=True)

    if args.checkpoint:
        print('LOADING PRETRAINED MODEL..')
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['state_dict']

        for key in list(state_dict.keys()):
            if 'queue' in key:
                del state_dict[key]

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)
    
    model = model.to(device)

    print("=" * 50)
    embeddings = get_smiles_embeddings(
        model, 
        smiles_list, 
        batch_size=config['batch_size_test']
    )
    
    df[args.embed_column] = [e.tolist() for e in embeddings]
    
    output_file = args.output_file or 'smiles_with_embeddings.csv'
    df.to_csv(output_file, index=False)
    print(f"CSV with embeddings saved to {output_file}")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='./Pretrain/checkpoint_SPMM.ckpt')
    parser.add_argument('--vocab_filename', default='./vocab_bpe_300.txt')
    parser.add_argument('--input_file', default='input.csv', help='Input CSV file with SMILES column')
    parser.add_argument('--output_file', default=None, help='Output CSV file path (default: smiles_with_embeddings.csv)')
    parser.add_argument('--smiles_column', default='Smiles', help='Name of the column containing SMILES strings')
    parser.add_argument('--embed_column', default='emb', help='Name for the embedding column in output CSV')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    args = parser.parse_args()

    config = {
        'embed_dim': 256,
        'batch_size_test': 64,
        'bert_config_text': './config_bert.json',
        'bert_config_property': './config_bert_property.json',
    }
    main(args, config)
