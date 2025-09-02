import sys
from pathlib import Path

# Ensure repo root is on sys.path so running this script directly finds the UTILS package
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from UTILS.insights import compute_top_claves_rechazo, build_top_claves_rechazo_figure
import pandas as pd

# Data de prueba
df = pd.DataFrame({
    'ClaveCatalogo':['A','B','A','C','D','A','B'],
    'DescripcionCatalogo':['dA','dB','dA','dC','dD','dA','dB'],
    'SubCategoria':['s1','s2','s1','s3','s4','s1','s2'],
    'Maquina':['M1','M2','M1','M3','M1','M2','M2'],
    'Turno':['T1','T1','T2','T2','T1','T2','T1'],
    'Pzas':[10,5,7,3,2,8,6]
})

for desg in ['Global','Máquina','Turno','Turno + Máquina']:
    print('\n--- Desglose:', desg)
    df_top, meta = compute_top_claves_rechazo(df, top_n=10, desglose=desg)
    print('df_top.shape =', df_top.shape)
    print(df_top)
    fig = build_top_claves_rechazo_figure(df_top, desg)
    print('fig traces =', len(fig.data))

print('\nTest completed')
