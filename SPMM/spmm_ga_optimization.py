import random

import numpy as np

import torch
from torch import nn
from transformers.tokenization_utils import Trie
from transformers import BertTokenizer, WordpieceTokenizer

from SPMM_models import SPMM, SPMMDownstreamPropertyPredictor
from calc_property import calculate_property
from d_pv2smiles_single import generate_with_property
from spmm_finetune_pdsc import ConfigDict


def surrogate_pdsc_predict(smiles, pv, model, tokenizer):
    text_input = tokenizer(smiles, padding='longest', truncation=True, max_length=100, return_tensors="pt")

    with torch.no_grad():
        pred_pdsc = model(property_original=pv.unsqueeze(0), text_input_ids=text_input.input_ids[:, 1:])

    return pred_pdsc


def generate_mol_by_property(properties):
    prop_positions = range(53)

    prop_mask = torch.ones(53)
    for pos in prop_positions:
        prop_mask[pos] = 0
    
    prop_input = torch.zeros(53)
    for pos in prop_positions:
        prop_input[pos] = properties[pos]

    samples = generate_with_property(model, prop_input, 1, prop_mask, stochastic=True, k=3)
    
    # In principle, we can control the strength of GA by selecting one of the top-k samples from the model
    sample = samples[0]

    return sample


def initialize_population(pop_size, num_dims, bounds):
    pop = torch.rand(pop_size, num_dims)
    for i in range(num_dims):
        pop[:, i] = (bounds[i][1] - bounds[i][0]) * pop[:, i] + bounds[i][0]

    return pop


def evaluate_individual(prop_vec, model, tokenizer):
    molecule = generate_mol_by_property(prop_vec)
    return surrogate_pdsc_predict(molecule, prop_vec, model, tokenizer) 


def tournament_selection(
    population: torch.Tensor,
    scores: torch.Tensor,
    k: int
) -> torch.Tensor:
    selected_indices = np.random.choice(len(population), k, replace=False)
    best_idx = selected_indices[np.argmax(scores[selected_indices])]

    # Return a copy of the property vector so it won't mutate the original accidentally
    return population[best_idx].clone()


def crossover(
    parent1: torch.Tensor,
    parent2: torch.Tensor,
    crossover_rate: float 
):
    child1 = parent1.clone()
    child2 = parent2.clone()

    for i in range(len(parent1)):
        if random.random() < crossover_rate:
            temp = child1[i].clone()
            child1[i] = child2[i]
            child2[i] = temp

    return child1, child2


def mutate(
    individual: torch.Tensor, 
    bounds: list,
    mutation_rate: float, 
    std_dev: float = 0.1,
):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            delta = random.gauss(0, std_dev) 
            individual[i] += delta

    for i in range(individual.shape[0]):
        individual[i].clamp_(bounds[i][0], bounds[i][1])


def get_surrogate_model():
    predictor_config = {
        'seed': 42,

        'data_path': '../data/data_new_spmm.csv',
        'pretrained_weights_path': '../SPMM_Checkpoint.ckpt',
        'vocab_path': './vocab_bpe_300.txt',

        'embed_dim': 256, # 256,
        'regression_head_hidden_dim': 64,

        'bert_config_text': './config_bert.json',
        'bert_config_property': './config_bert_property.json',
    }
    predictor_config = ConfigDict(predictor_config)

    model = SPMMDownstreamPropertyPredictor(predictor_config, property_width=768)
    model.text_encoder.cls = nn.Identity()
    model.property_encoder.cls = nn.Identity()

    model.load_state_dict(torch.load('../pdsc_predictor.ckpt', map_location='cpu'))

    model.eval()

    tokenizer = BertTokenizer(vocab_file=predictor_config.vocab_path, do_lower_case=False, do_basic_tokenize=False)
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token, max_input_chars_per_word=250)

    return model, tokenizer


def run_ga(start_props=None):
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    POP_SIZE = 50
    NUM_PROPS = 53
    NUM_GENERATIONS = 100
    MUTATION_RATE = 0.2
    CROSSOVER_RATE = 0.5
    PROPERTY_RANGE = [(0.0, 1.0) for _ in range(NUM_PROPS)]

    model, tokenizer = get_surrogate_model()

    if start_props is None:
        population = initialize_population(POP_SIZE, NUM_PROPS, PROPERTY_RANGE)
    else:
        population = start_props.unsqueeze(0).expand(POP_SIZE, -1)

    scores = torch.tensor([evaluate_individual(ind, model, tokenizer) for ind in population])
    
    best_score = -float('inf')
    best_vector = None

    for gen in range(NUM_GENERATIONS):
        # Track generation's best
        gen_best_idx = torch.argmax(scores)
        gen_best_score = scores[gen_best_idx].item()
        if gen_best_score > best_score:
            best_score = gen_best_score
            best_vector = population[gen_best_idx].clone()

        print(f"[Gen {gen}] Best in population: {gen_best_score:.4f}, Overall best: {best_score:.4f}")

        # Create new population
        new_population = []
        while len(new_population) < POP_SIZE:
            # Selection
            parent1 = tournament_selection(population, scores, k=5)
            parent2 = tournament_selection(population, scores, k=5)

            # Crossover
            child1, child2 = crossover(parent1, parent2, CROSSOVER_RATE)

            # Mutation
            mutate(child1, PROPERTY_RANGE, MUTATION_RATE)
            mutate(child2, PROPERTY_RANGE, MUTATION_RATE)
            new_population.append(child1)
            new_population.append(child2)

        # Make sure population is the correct size
        new_population = new_population[:POP_SIZE]
        population = torch.stack(new_population, dim=0)
        
        # Evaluate new individuals
        try:
            scores = torch.tensor([evaluate_individual(ind, model, tokenizer) for ind in population])
        except:
            print(population)
            raise ValueError
    
    print("===== GA Finished =====")
    print("Best property vector found:\n", best_vector)
    print(f"Best predicted property score: {best_score:.4f}")

    # Decode final best with the Transformer
    best_mol = generate_mol_by_property(best_vector)
    print("Example best molecule:", best_mol)


if __name__ == '__main__':
    config = {
        'embed_dim': 256,
        'bert_config_text': './config_bert.json',
        'bert_config_property': './config_bert_property.json',
    }

    tokenizer = BertTokenizer(vocab_file='./vocab_bpe_300.txt', do_lower_case=False, do_basic_tokenize=False)
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token, max_input_chars_per_word=250)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SPMM(config=config, tokenizer=tokenizer, no_train=True)

    checkpoint_path = '../SPMM_Checkpoint.ckpt'
    if checkpoint_path is not None:
        print('LOADING PRETRAINED MODEL..')
    
        with torch.serialization.safe_globals([BertTokenizer, Trie, WordpieceTokenizer]):
            state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']

        for key in list(state_dict.keys()):
            if 'word_embeddings' in key and 'property_encoder' in key:
                del state_dict[key]
            if 'queue' in key:
                del state_dict[key]

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % checkpoint_path)

    model = model.to(device)

    start_smiles = 'C=C'
    properties = calculate_property(start_smiles)

    run_ga(start_props=None)