import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)
df = pd.read_csv('padnew.csv')
conso = df[df['Labels'].str.contains("consolidation", na=False)]
infil = df[df['Labels'].str.contains("infiltrates", na=False)]
atel = df[df['Labels'].str.contains("atelectasis", na=False)]
ground = df[df['Labels'].str.contains("ground glass pattern", na=False)]
resp = df[df['Labels'].str.contains("respiratory distress", na=False)]
pneu = df[df['Labels'].str.contains("pneumonia", na=False)]
normal = df[df['Labels'].str.contains("normal", na=False)]
df1 = pd.concat([conso, infil, atel, ground, resp, pneu, normal])

for index, row in df1.iterrows():
    a = ""
    if row['Labels'].find("consolidation")!=-1:
        a = "consolidation,"
        
    if row['Labels'].find("infiltrates")!=-1:
        a+="infiltrates,"
    if row['Labels'].find("atelectasis")!=-1:
        a+="atelectasis,"
    if row['Labels'].find("ground glass pattern")!=-1:
        a+="ground glass pattern,"
    if row['Labels'].find("respiratory distress")!=-1:
        a+="respiratory distress,"
    if row['Labels'].find("pneumonia")!=-1:
        a+="pneumonia,"

    if row['Labels'].find("normal")!=-1:
        a+="normal,"
    #print(l)
    df1.at[index, 'Labels'] = a[:-1]
    #print(row['Labels'])
#df1.to_csv('padnew.csv')
#print(df1['Labels'].head(10))

dick = {'loc left lower lobe' : 'L',
'loc right' : 'R',
'loc basal bilateral': 'B',
'loc fissure' : 'R',
'loc left' : 'L',
'loc bilateral costophrenic angle' : 'B',
'loc left' : 'L',
'loc lingula' : 'L',
'loc retrocardiac': 'L',
'loc left upper lobe': 'L',
'loc right upper lobe': 'R',
'loc bilateral': 'B',
'loc left upper lobe' : 'L',
'loc middle lobe' : 'R',
'loc bilateral costophrenic angle' : 'B',
'loc right lower lobe': 'R',
'loc left lower lobe' : 'L',
'loc hilar bilateral': 'B',
'loc superior cave vein': 'R',
'loc lingula': 'L',
'loc diffuse bilateral' : 'B',
'loc right costophrenic angle' : 'R',
'loc middle lobe': 'R',
'loc right': 'R',
'loc left costophrenic angle': 'L',
'loc hilar bilateral': 'B',
'loc right lower lobe' : 'R',
'loc retrocardiac': 'L',
'loc right upper lobe':'R',
'loc diffuse bilateral': 'B',
'loc right costophrenic angle': 'R',
'loc basal bilateral': 'B',
'loc bilateral': 'B'
    }

for index, row in df.iterrows():
    a=''
    b=''
    for key, value in dick.items():
        if row['Localizations'].find(key)!=-1:
            a = a+value+','
    if ('L' in a and 'R' in a):
        b = 'B'
    elif 'B' in a:
        b = 'B'
    elif 'L' in a:
        b = 'L'
    elif 'R' in a:
        b = 'R'
    df.at[index, 'Localizations'] = b
    
df.to_csv('test.csv')
