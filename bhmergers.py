import numpy as np
import pynbody

def get_mergers_by_id(bhiord, mdata):
	# return mdata['IDeat'][(mdata['ID']==bhiord)], mdata['step'][(mdata['ID']==bhiord)]
	match = np.where((mdata['ID1'] == bhiord) & (mdata['merge_mass_1'] >= 1e6) & (mdata['merge_mass_2'] >= 1e6) & (
				np.minimum(mdata['tform1'], mdata['tform2']) > 0))[0]
	match_all = np.where(mdata['ID1'] == bhiord)[0]
	# strict = BHs formed at initial separations greater than a kpc
	return mdata['ID2'][match], mdata['time'][match], mdata['tform1'][match], mdata['tform2'][match], \
	       mdata['init_dist'][match], mdata['ID2'][match_all]


def get_all_mergers(bhiord, mdata):
	bhlist = list([bhiord])
	bhlist_new = list([])
	id_list_all = list([])
	id_list = list([])
	id_list_strict = list([])
	time_list = list([])
	dist_list = list([])
	tform1_list = list([])
	tform2_list = list([])
	while len(bhlist) > 0:
		for i in range(len(bhlist)):
			bhlist_part, time_part, tform1_part, tform2_part, dist_part, id_all_part = get_mergers_by_id(bhlist[i],
			                                                                                             mdata)
			bhlist_new.extend(bhlist_part)
			id_list.extend(bhlist_part)
			time_list.extend(time_part)
			tform1_list.extend(tform1_part)
			tform2_list.extend(tform2_part)
			id_list_all.extend(id_all_part)
			dist_list.extend(dist_part)
		bhlist = bhlist_new
		bhlist_new = list([])
	time_list = pynbody.array.SimArray(time_list, 'Gyr')
	tform1_list = pynbody.array.SimArray(tform1_list, 'Gyr')
	tform2_list = pynbody.array.SimArray(tform2_list, 'Gyr')
	dist_list = pynbody.array.SimArray(dist_list, 'kpc')

	return np.array(id_list), time_list, tform1_list, tform2_list, dist_list, np.array(id_list_all)


def collect_all_bh_mergers(step, mdata):
	halo_id_cen, bhid_cen = step.gather_property('halo_number()',
	                                 'link(BH_central, BH_mdot, "max", BH_central_distance<1).halo_number()')

	halo_id_cen, bhid_any = step.gather_property('halo_number()',
	                        'link(BH_central, BH_mdot, "max").halo_number()')

	tot_bhids = np.append(bhid_cen, bhid_any)
	tot_bhids = np.unique(tot_bhids)
	nmerge = np.ones(len(tot_bhids)) * -1
	nmerge_all = np.ones(len(tot_bhids)) * -1
	tmerge = list([])
	iord_merge = list([])
	iord_merge_all = list([])
	hid_bh = tot_bhids
	tlast = np.ones(len(tot_bhids)) * -1
	tform1 = list([])
	tform2 = list([])
	init_dist = list([])

	for i in range(len(tot_bhids)):
		frac = i / len(tot_bhids)
		if frac % 0.1 < 1 / len(tot_bhids): print(frac)
		prog_id, tm, tf1, tf2, dd, prog_all = get_all_mergers(tot_bhids[i], mdata)
		nmerge[i] = len(prog_id)
		nmerge_all[i] = len(prog_all)
		tmerge.append(tm)
		iord_merge.append(prog_id)
		iord_merge_all.append(prog_all)
		tform1.append(tf1)
		tform2.append(tf2)
		init_dist.append(dd)
		if len(prog_id) > 0:
			tlast[i] = tm.max()

	return nmerge, tmerge, iord_merge, iord_merge_all, tlast, hid_bh, tform1, tform2, nmerge_all

