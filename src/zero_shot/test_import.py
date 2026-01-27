import rootutils
rootutils.setup_root(__file__, indicator='setup.py', pythonpath=True)

import utils
print(dir(utils))