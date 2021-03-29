from asr.test import test_findmoire
from ase.db import connect

db = connect('/home/niflheim2/steame/moire/utils/c2db.db/')
uid_a = 'MoS2-b3b4685fb6e1' 
uid_b = 'WS2-64090c9845f8' 
supc = test_findmoire.main(uid_a=uid_a, uid_b=uid_b)
print(supc.pair1)
info_a = db.get(uid=uid_a)
info_b = db.get(uid=uid_b)


