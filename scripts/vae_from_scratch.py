#learn a vae from scratch

from vae_dimension_reduce import VaeDimensionReducer

nrows = 1000
vd = VaeDimensionReducer()
vd.load_from_csv(nrows=nrows)

vd.learn_reducer(latentD = 20, savefreq=10, epochs=200)


