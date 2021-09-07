import os
import logging
import argparse
import numpy as np
import pyqg

logging.basicConfig(level = logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, help="output directory",
                    required=True)
parser.add_argument("--num_cores", type=int, help="number of CPU cores to use",
                    default=16)
parser.add_argument("--nx", type=int, help="horizional resolution",
                    default=512)                  
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
logging.info(f"Output saving to {args.output_dir}")

year = 24*60*60*360.
day = 24*60*60.
m = pyqg.QGModel(nx=args.nx, dt=1600, tmax=10*year, twrite=30*day, tavestart=5*year, ntd=args.num_cores)
for t in m.run_with_snapshots(tsnapstart=90*day, tsnapint=5*day):
    q = m.q
    num_days = int(t/day)
    ke = m._calc_ke()
    cfl = m._calc_cfl()
    logging.info('Step: %i, Time (day): %d, KE: %3.2e, CFL: %4.3f', m.tc, num_days, ke, cfl)
    if cfl >= 1:
        logging.error('CFL condition violated')
        raise RuntimeError('CFL condition violated: reducing timestep')

    output_path = os.path.join(args.output_dir, str(num_days) + ".npy")
    np.save(output_path, q)