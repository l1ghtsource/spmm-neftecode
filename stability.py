from rdkit import Chem
from rdkit.Chem import AllChem

def check_conformers(smiles):
    """
    Принимает SMILES-строку, генерирует 30 конформаций, оптимизирует их с помощью MMFF94 и 
    определяет, сколько из них имеют энергию, близкую к минимальной (в пределах одной пятой 
    от разброса энергий). Если таких конформаций больше 13, функция возвращает True, иначе False.

    Parameters:
    -----------
    smiles : str
        SMILES-представление молекулы.

    Returns:
    --------
    bool
        True, если число конформаций с энергией ≤ (min_energy + (range/5)) больше 13, иначе False.
    """да
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Неверная SMILES-строка")
    mol = Chem.AddHs(mol)
    
    conformer_ids = AllChem.EmbedMultipleConfs(
        mol, numConfs=30,
        useExpTorsionAnglePrefs=True,
        useBasicKnowledge=True
    )
    
    energies = []
    props = AllChem.MMFFGetMoleculeProperties(mol)
    for conf_id in conformer_ids:
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
        ff.Minimize()
        energy = ff.CalcEnergy()
        energies.append(energy)
    
    min_energy = min(energies)
    max_energy = max(energies)
    
    energy_range = max_energy - min_energy
    threshold = min_energy + energy_range / 5.0
    
    close_confs_count = sum(1 for e in energies if e <= threshold)
    
    return close_confs_count > 13

if __name__ == "__main__":
    test_smiles = input('Введите SMILES: ') # "CC(C)(C)C1=C(C(=CC(=C1O)C(C)(C)C)C(C)(C)C)OCC(C)(C)C2=C(C(=CC(=C2O)C(C)(C)C)C(C)(C)C)OCC(C)(C)C3=C(C(=CC(=C3O)C(C)(C)C)C(C)(C)C)O"
    result = check_conformers(test_smiles)
    print("Результат:", result)