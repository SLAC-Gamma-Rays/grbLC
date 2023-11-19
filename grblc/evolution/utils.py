import numpy as np

def _merge_dicts(dict1, dict2):
	# assumes dict1 and dict2 hold one dimensional arrays
	final = {k: None for k in np.append(list(dict1.keys()), list(dict2.keys()))}

	for k in final:
		arr1 = dict1.get(k, np.array([]))
		arr2 = dict2.get(k, np.array([]))
		final[k] = np.append(arr1, arr2)

	return final
