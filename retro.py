import streamlit as st
import time
from rxn4chemistry import RXN4ChemistryWrapper
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import io
from PIL import Image

st.set_page_config(
    page_title="Retrosynthesis Predictor",
    page_icon="🧪",
    layout="wide"
)

st.markdown("""
<style>
    .loading-spinner {
        text-align: center;
        margin: 30px 0;
    }
    .reaction-container {
        border: 1px solid #eee;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .result-header {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧪 Retrosynthesis Prediction Tool")
st.markdown("Enter a molecule SMILES to predict possible synthetic routes using IBM RXN.")

def collect_reactions(tree):
    reactions = []
    if 'children' in tree and len(tree['children']):
        reactions.append(
            AllChem.ReactionFromSmarts('{}>>{}'.format(
                '.'.join([node['smiles'] for node in tree['children']]),
                tree['smiles']
            ), useSmiles=True)
        )
    for node in tree['children']:
        reactions.extend(collect_reactions(node))
    return reactions

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your RXN4Chemistry API key", type="password")
    
    st.markdown("---")
    st.markdown("### Example SMILES")
    examples = {
        "Diphenylamine": "C1=CC=C(C=C1)NC2=CC=CC=C2",
        "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "Paracetamol": "CC(=O)Nc1ccc(O)cc1"
    }
    
    for name, smiles in examples.items():
        if st.button(name):
            st.session_state.smiles_input = smiles
            st.rerun()
            
    st.markdown("---")
    st.info("⚠️ **Important Notice:**\nAccessing this service from Russian IP addresses is not possible. Please use a VPN if connecting from Russia.")

if 'smiles_input' not in st.session_state:
    st.session_state.smiles_input = "C1=CC=C(C=C1)NC2=CC=CC=C2"

smiles_input = st.text_input("Input SMILES notation", value=st.session_state.smiles_input)
st.session_state.smiles_input = smiles_input

try:
    mol = Chem.MolFromSmiles(smiles_input)
    if mol:
        st.subheader("Target Molecule")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            img = Draw.MolToImage(mol, size=(250, 200))
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            st.image(img_bytes, width=250)
        
        with col2:
            st.markdown(f"**SMILES:** `{smiles_input}`")
            st.markdown(f"**Formula:** {Chem.rdMolDescriptors.CalcMolFormula(mol)}")
except Exception as e:
    st.error(f"Error displaying molecule: {str(e)}")

if st.button("🔬 Predict Retrosynthesis"):
    if not api_key:
        st.error("Please enter your RXN4Chemistry API key in the sidebar")
    elif not smiles_input:
        st.error("Please enter a valid SMILES notation")
    else:
        try:
            rxn4chemistry_wrapper = RXN4ChemistryWrapper(api_key=api_key)
            rxn4chemistry_wrapper.create_project(f"streamlit-{int(time.time())}")
            print(rxn4chemistry_wrapper.project_id)
            
            with st.spinner("Submitting retrosynthesis prediction request..."):
                response = rxn4chemistry_wrapper.predict_automatic_retrosynthesis(
                    product=smiles_input, ai_model='2019-09-12'
                )
                prediction_id = response['prediction_id']
                st.success(f"Prediction request submitted successfully!")
            
            status = "PENDING"
            results = None
            
            spinner_placeholder = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            max_attempts = 50
            attempt = 0
            
            while status != "SUCCESS" and attempt < max_attempts:
                dots = "." * (attempt % 4 + 1)
                spinner_placeholder.markdown(f"""
                <div class="loading-spinner">
                    <h3>🧪 RXN is thinking{dots}</h3>
                    <p>Analyzing possible synthesis routes...</p>
                </div>
                """, unsafe_allow_html=True)
                
                results = rxn4chemistry_wrapper.get_predict_automatic_retrosynthesis_results(prediction_id)
                status = results.get('status', 'PENDING')
                
                progress = min(attempt / max_attempts, 0.95)
                progress_bar.progress(progress)
                status_text.text(f"Status: {status} (Step {attempt+1})")
                
                if status == "SUCCESS":
                    break
                
                time.sleep(0.5)
                attempt += 1
            
            progress_bar.progress(1.0)
            
            if status != "SUCCESS":
                spinner_placeholder.empty()
                st.error(f"Prediction did not complete in time. Current status: {status}")
                st.stop()
            
            spinner_placeholder.empty()
            progress_bar.empty()
            status_text.empty()
            
            if results and 'retrosynthetic_paths' in results:
                st.markdown(f"""
                <div class="result-header" style="background-color: #e7f5ea;">
                    <h2>✅ Retrosynthesis Complete!</h2>
                    <p>Found {len(results['retrosynthetic_paths'])} potential synthetic pathway(s)</p>
                </div>
                """, unsafe_allow_html=True)
                
                if len(results['retrosynthetic_paths']) > 0:
                    path_tabs = st.tabs([f"Path {i+1} (Confidence: {path['confidence']:.2f})" 
                                         for i, path in enumerate(results['retrosynthetic_paths'])])
                    
                    for i, (path, tab) in enumerate(zip(results['retrosynthetic_paths'], path_tabs)):
                        with tab:
                            reactions = collect_reactions(path)
                            
                            if not reactions:
                                st.warning("No reactions found in this path.")
                            else:
                                st.write(f"**Total reactions:** {len(reactions)}")
                                
                                for j, reaction in enumerate(reactions):
                                    st.write(f"### Reaction {j+1}")
                                    
                                    with st.container():
                                        st.markdown('<div class="reaction-container">', unsafe_allow_html=True)
                                        
                                        img = Draw.ReactionToImage(reaction, subImgSize=(300, 200))
                                        img_bytes = io.BytesIO()
                                        img.save(img_bytes, format='PNG')
                                        img_bytes.seek(0)
                                        
                                        st.image(img_bytes, use_container_width =True)
                                        
                                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("No viable synthetic pathways were found.")
            else:
                st.error("No retrosynthetic paths found in the results.")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
