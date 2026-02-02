## Data and Code Repository for: <br/>Performance and Reliability of LLMs and AI Agents for Deep Learning Code Generation in Ophthalmology </br>

This repository contains data splits, LLM- and AI agent-generated code, and classifier results for the aforementioned paper.   
Please contact thad.lo.seri@gmail.com if you have any questions.

### dataset_splits
- Contains dataset splits used for training datasets (APTOS-2019, PAPILA, UKBB)
- All images in the external validation datasets (MESSIDOR2, VEIRC, ADAM) are used for testing, except for VEIRC images labelled as suspected glaucoma 
- For the colour fundus photographs, please refer to the official sources:
    - Diabetic retinopathy: [APTOS-2019](https://www.kaggle.com/competitions/aptos2019-blindness-detection) | [MESSIDOR2](https://www.adcis.net/en/third-party/messidor2/)
    - Glaucoma: [PAPILA](https://figshare.com/articles/dataset/PAPILA/14798004?file=35013982) | [VEIRC](https://github.com/ProfMKD/Glaucoma-dataset)
    - Age-related macular degeneration: [UKBB](https://www.ukbiobank.ac.uk/) | [ADAM](https://amd.grand-challenge.org/)

### main
- Contains LLM- and AI agent-generated code, as well as trained classifier results for the main set of results
- Organized by classifier architecture -> input prompt type -> LLM or AI agent -> contents

### sensitivity analysis
- Contains LLM- and AI agent-generated code, as well as trained classifier results for the additional sensitivity analysis (two independent repeats)
- Organized by classifier architecture -> input prompt type -> LLM or AI agent -> iteration -> contents