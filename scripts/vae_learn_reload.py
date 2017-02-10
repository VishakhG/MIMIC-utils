#Continue learning a VAE from saved weights

from vae_dimension_reduce import VaeDimensionReducer
vd = VaeDimensionReducer(save_prefix = "48_hr_all_100D")
vd.load_from_csv()
pfile = "/data/ml2/vishakh/vae/out/48_hr_all_100D_25epchparams.pkl"
reloadFile = "/data/ml2/vishakh/vae/out/48_hr_all_100D_25epchparams-params.npz"
vd.learn_reducer(reloadFile = reloadFile, pfile=pfile, latentD =100, savefreq=10)
