import tempfile
import pathlib
import time
import numpy as np
from cosmosis.runtime import Inifile, LikelihoodPipeline

ROOT = pathlib.Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as directory:
    d = pathlib.Path(directory)
    (d/'values.ini').write_text('[cosmological_parameters]\nombh2=.022\nomch2=.11 .12 .13\nh0=.675\ntau=.06\nn_s=.965\nA_s=2.1e-9\nomega_k=0\n[nuisance]\nx=-1 0 1\n')
    (d/'like.py').write_text('def setup(options): return {}\ndef execute(block, config):\n x=block["nuisance","x"]\n p=block["matter_power_lin","p_k"]\n block["likelihoods","probe_like"]=-0.5*x*x-float(p[0,0])*1e-6\n return 0\n')
    reference = None
    timings = {False: [], True: []}
    for repetition in range(3):
        fs = True
        for reuse in ((False, True) if repetition % 2 == 0 else (True, False)):
            ini=Inifile({'pipeline':{'modules':'camb probe','values':str(d/'values.ini'),'likelihoods':'probe','fast_slow':str(fs),'quiet':'T'},'camb':{'file':str(ROOT/'boltzmann/camb/camb_interface.py'),'mode':'power','nonlinear':'none','nk':'20','nz':'6','reuse_computation':str(reuse)},'probe':{'file':str(d/'like.py')}})
            pipeline=LikelihoodPipeline(ini)
            if fs: pipeline.setup_fast_subspaces()
            cfg=pipeline.modules[0].data[1]
            cfg['_reuse_hits']=cfg['_reuse_misses']=0
            cfg['_reuse_cache'].clear()
            outputs=[]
            start=time.perf_counter()
            for density,x in [(0.120+0.001*i, 0.1*j) for j in range(3) for i in range(4)]:
                values=[density if p.section=='cosmological_parameters' else x for p in pipeline.varied_params]
                block=pipeline.run_parameters(values)
                assert block is not None
                outputs.append(block['matter_power_lin','p_k'].copy())
            elapsed=time.perf_counter()-start
            timings[reuse].append(elapsed)
            if reference is None: reference=outputs
            eps=max(float(np.max(np.abs(a-b)/np.maximum(np.abs(b),1e-30))) for a,b in zip(outputs,reference))
            assert eps <= 1e-12
            print('RESULT',fs,reuse,elapsed,cfg['_reuse_hits'],cfg['_reuse_misses'],eps,flush=True)
    for reuse, samples in timings.items():
        print('SUMMARY', reuse, 'times', samples, 'median', np.median(samples),
              'std', np.std(samples, ddof=1), flush=True)
    print('SPEEDUP', np.median(timings[False])/np.median(timings[True]), flush=True)

