import wandb
api = wandb.Api()

# --- Downloading the model ---

# Runs mit korrektem Entity-Namen
'''
runs = api.runs('anton-freitaeger-universit-t-m-nster/test-000')
print('Verfügbare Runs:')
for run in runs:
    print(f'  ID: {run.id}, Name: {run.name}, State: {run.state}')
'''

run = api.run('anton-freitaeger-universit-t-m-nster/test-000/l1cydkoe') # <--- CHANGE THIS URL runs/xxx/files

# trained_model.pt herunterladen
if 'tmp964yzj69/trained_model.pt' in [f.name for f in run.files()]:     # <--- CHANGE THIS
    run.file('tmp964yzj69/trained_model.pt').download()                 # <--- CHANGE THIS
    print('trained_model.pt erfolgreich heruntergeladen!')
else:
    print('trained_model.pt nicht gefunden')
