import sacc
import numpy as np
import matplotlib.pyplot as plt

r = 2

def reduce(infile, outfile):
    s = sacc.Sacc.load_fits(infile)

    for name, tracer in list(s.tracers.items()):
        print(name, tracer)
        z = tracer.z
        nz = tracer.nz
        # downsample n(z)
        nred = len(z) // r
        zred = []
        nzred = []
        for i in range(nred):
            zred.append(np.mean(z[i * r:(i + 1) * r]))
            nzred.append(np.mean(nz[i * r:(i + 1) * r]))
        zred = np.array(zred)
        nzred = np.array(nzred)
        cut = zred < 4
        zred = zred[cut]
        nzred = nzred[cut]
        new_tracer = sacc.tracers.NZTracer(name, zred, nzred)
        s.tracers[name] = new_tracer


    for name, tracer in list(s.tracers.items()):
        print(name, tracer)
        print(tracer.z.shape)
        plt.plot(tracer.z, tracer.nz, label=name)
    plt.xlabel("z")
    plt.ylabel("n(z)")
    plt.legend()
    plt.show()

    s.save_fits(outfile, overwrite=True)


if __name__ == "__main__":
    infile = "/Users/jzuntz/src/euclid/txpipe/TXPipe/data/euclid-rr2/outputs-2_1-lensmc/summary_statistics_real.sacc"
    outfile = f"reduced_{r}_euclid_rr2_2.1_summary_statistics_real.sacc"
    reduce(infile, outfile)

    infile = "/Users/jzuntz/src/euclid/txpipe/TXPipe/data/euclid-rr2/outputs-2_1-lensmc/summary_statistics_fourier.sacc"
    outfile = f"reduced_{r}_euclid_rr2_2.1_summary_statistics_fourier.sacc"
    reduce(infile, outfile)
