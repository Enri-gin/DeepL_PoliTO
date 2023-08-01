import subprocess
# from google.colab import drive
import os 

print('Folder corrente', os.getcwd())
os.chdir('..')
print('Folder corrente', os.getcwd())


#repository_url = 'https://github.com/Enri-gin/MLandDL_polito.git'

#subprocess.call(['git', 'clone', repository_url])
#print('--Cartella git clonata')

# Rimuovi la directory "sample_data"
subprocess.run(['rm', '-r', 'sample_data'])

os.chdir('..')


# Esegui lo script di download COCO dataset
subprocess.run(['bash', 'DownloadVWW/scripts/download_mscoco.sh', 'DownloadVWW/scripts'])



print('--COCO dataset scaricato')

# Copia i contenuti di train2014 e val2014 in all2014
subprocess.run(['cp', '-r', 'DownloadVWW/scripts/train2014/.', 'DownloadVWW/scripts/all2014'])
subprocess.run(['cp', '-r', 'DownloadVWW/scripts/val2014/.', 'DownloadVWW/scripts/all2014'])

print(r'--contenuti da train2014/val2014 copiati in all2014')

# Esegui lo script create_coco_train_minival_split.py
subprocess.run([
    'python',
    'DownloadVWW/scripts/create_coco_train_minival_split.py',
    '--train_annotations_file=DownloadVWW/scripts/annotations/instances_train2014.json',
    '--val_annotations_file=DownloadVWW/scripts/annotations/instances_val2014.json',
    '--output_dir=DownloadVWW/scripts/annotations/'
])
print('--Fatto lo split tra train e val')

# Esegui lo script create_visualwakewords_annotations.py
subprocess.run([
    'python',
    'DownloadVWW/scripts/create_visualwakewords_annotations.py',
    '--train_annotations_file=DownloadVWW/scripts/annotations/instances_maxitrain.json',
    '--val_annotations_file=DownloadVWW/scripts/annotations/instances_minival.json',
    '--output_dir=DownloadVWW/scripts/annotations/',
    '--threshold=0.005',
    '--foreground_class=person'
])
print('--Create le annotation')

# Rimuovi le directory train2014, val2014 ed examples
subprocess.run(['rm', '-r', 'DownloadVWW/scripts/train2014'])
subprocess.run(['rm', '-r', 'DownloadVWW/scripts/val2014'])
subprocess.run(['rm', '-r', 'DownloadVWW/examples'])