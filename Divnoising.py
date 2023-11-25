signal=np.mean(observations)#avg noise clean signal of all the images
histogram = histNoiseModel.createHistogram(bins, minVal, maxVal, observation,signal)
min_signal=np.min(signal)
max_signal=np.max(signal)
gaussianMixtureNoiseModel = GaussianMixtureNoiseModel(min_signal = min_signal,
                                                                                max_signal =max_signal,
                                                                                path=path, weight = None,
                                                                                n_gaussian = n_gaussian,
                                                                                n_coeff = n_coeff,
                                                                                min_sigma = 50,
                                                                                device = device)
gaussianMixtureNoiseModel.train(signal, observation, batchSize = 250000, n_epochs = 2000,
                                learning_rate=0.1, name = nameGMMNoiseModel)