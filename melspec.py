MEL = []
sum = 0
for i in range(X.shape[0]):
    key = sample_key[i]
    key_pref = key[:3]
    y, sr = lb.load(DATA_DIR + 'audio/' + key_pref + '/' + key + '.ogg')
    S = lb.feature.melspectrogram(y=y, sr=sr)
    S_dB = lb.power_to_db(S, ref=0)
    MEL.append(S_dB[:,:430])
    
MEL_S = np.asarray(MEL)
print('Mel has shape: ' + str(MEL_S.shape))

# TODO SAVE WITHOUT X

np.savez('openmic-test-delete.npz', MEL = X, Y_true=Y_true, Y_mask=Y_mask, sample_key=sample_key)

np.savez_compressed('openmic-mel-only.npz', MEL = MEL_S)
print('OpenMIC keys: ' + str(list(OPENMIC_2.keys())))