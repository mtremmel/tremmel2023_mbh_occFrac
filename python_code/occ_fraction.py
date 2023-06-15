default_bins = [7, 7.25, 7.5, 7.75, 8, 8.25, 8.5, 8.75, 9, 9.5, 10, 10.5, 11.5]
default_time_bins = [0,2,4,6,8,10,12,14]
import numpy as np
from .util import identify_quench
from .util import identify_sats_geha
from .util import binomial_errors
from .util import calc_xray_lum
from .util import calc_bol_corr_xray
from .util import find_closest_host
import pynbody
import Simpy
import tangos as db

def collect_host_info(timestep_data, simstep):

	print("gathering timestep properties...")
	hid, cen, r200, mvir, mstar = simstep.calculate_all('halo_number()', 'shrink_center', 'radius(200)', 'Mvir', 'Mstar')
	print("selecting close systems...")
	mhost, Dhost, hid_host, mclose, Dclose, hid_close = find_closest_host(cen, r200, mstar, mvir, hid, scale=1)

	dist_host = np.ones(len(timestep_data['hid']))*-1
	mvir_host = np.ones(len(timestep_data['hid']))*-1
	dist_close = np.ones(len(timestep_data['hid'])) * -1
	mvir_close = np.ones(len(timestep_data['hid'])) * -1

	for i in range(len(timestep_data['hid'])):
		hid_target = timestep_data['hid'][i]
		ind_target = np.where(hid==hid_target)[0]
		if len(ind_target)==0:
			continue
		if len(ind_target)>1:
			raise RuntimeError("found multiple entries in database for", hid_target)
		dist_host[i] = Dhost[ind_target[0]]
		mvir_host[i] = mhost[ind_target[0]]
		dist_close[i] = Dclose[ind_target[0]]
		mvir_close[i] = mclose[ind_target[0]]

	timestep_data['dist_host'] = dist_host
	timestep_data['mvir_host'] = mvir_host
	timestep_data['dist_close'] = dist_close
	timestep_data['mvir_close'] = mvir_close

	return

def get_galaxy_bh_data_cluster(step, calc_formation_times=False, star_corr=0.6, verbose=True, t50min=1e7,t50max=1e9):
	print("getting database data for step", step)
	if calc_formation_times:
		hid_all, mstar, mvir, cen, r200, contam, t50, t80 = step.gather_property('halo_number()', 'Mstar', 'Mvir',
		                                                                 'shrink_center',
		                                                                 'radius(200)', 'contamination_fraction','latest().t50()',
		                                                                 'latest().t80()')
	else:
		hid_all, mstar, mvir, cen, r200, contam = step.gather_property('halo_number()', 'Mstar', 'Mvir',
		                                                       'shrink_center',
		                                                       'radius(200)', 'contamination_fraction')

	hid_bh, mstar_bh, mvir_bh, mbh, mdot, mdot_mean, bhid = step.gather_property('halo_number()', 'Mstar',
	                                                                             'Mvir',
	                                                                             'link(BH_central, BH_mdot, "max", BH_central_distance<1).BH_mass',
	                                                                             'link(BH_central, BH_mdot, "max", BH_central_distance<1).BH_mdot',
	                                                                             'link(BH_central, BH_mdot, "max", BH_central_distance<1).BH_mdot_ave',
	                                                                             'link(BH_central, BH_mdot, "max", BH_central_distance<1).halo_number()')

	hid_bh_any, mstar_bh_any, mvir_bh_any, mbh_any, mdot_any, mdot_mean_any, bhid_any = step.gather_property(
		'halo_number()',
		'Mstar', 'Mvir',
		'link(BH_central, BH_mdot, "max").BH_mass',
		'link(BH_central, BH_mdot, "max").BH_mdot',
		'link(BH_central, BH_mdot, "max").BH_mdot_ave',
		'link(BH_central, BH_mdot, "max").halo_number()')

	print("selecting resolved halos, crossmatching halos with black holes")

	darr = cen - cen[(hid_all == 1)]
	print(darr)
	Simpy.util.wrap(darr, boxsize=50e3, scale=1. / (step.redshift + 1))
	D = np.sqrt(np.sum(darr ** 2, axis=1))
	use = np.where((D < 2000) & (contam < 0.05) & (mvir > 3e8))[0]
	hid_all = hid_all[use]
	mstar = mstar[use]
	mvir = mvir[use]
	D = D[use]

	use_bh = np.where(np.in1d(hid_bh, hid_all))[0]
	hid_bh = hid_bh[use_bh]
	bhid = bhid[use_bh]
	mbh = mbh[use_bh]
	mdot = mdot[use_bh]
	mdot_mean = mdot_mean[use_bh]

	# cross check with resolved halos
	use_bh_any = np.where(np.in1d(hid_bh_any, hid_all))[0]
	hid_bh_any = hid_bh_any[use_bh_any]
	bhid_any = bhid_any[use_bh_any]
	mbh_any = mbh_any[use_bh_any]
	mdot_any = mdot_any[use_bh_any]
	mdot_mean_any = mdot_mean_any[use_bh_any]

	has_bh_cen = np.zeros(len(hid_all))
	has_bh_any = np.zeros(len(hid_all))
	has_bh_cen[(np.in1d(hid_all, hid_bh))] = 1
	has_bh_any[(np.in1d(hid_all, hid_bh_any))] = 1

	cen_bh_id = np.ones(len(hid_all)) * -1
	any_bh_id = np.ones(len(hid_all)) * -1
	cen_bh_mass = np.ones(len(hid_all)) * -1
	any_bh_mass = np.ones(len(hid_all)) * -1
	cen_bh_mdot = np.ones(len(hid_all)) * -1
	any_bh_mdot = np.ones(len(hid_all)) * -1
	cen_bh_mdot_mean = np.ones(len(hid_all)) * -1
	any_bh_mdot_mean = np.ones(len(hid_all)) * -1

	if calc_formation_times:
		t50_star_trace = np.ones(len(hid_all)) * -1
		t80_star_trace = np.ones(len(hid_all)) * -1
		t50_mvir_trace = np.ones(len(hid_all)) * -1
		t80_mvir_trace = np.ones(len(hid_all)) * -1
		mvir_max = np.ones(len(hid_all)) * -1
		mstar_max = np.ones(len(hid_all)) * -1
		print("collecting time evolution data...")

	cnt = 0
	print("collecting black hole data...")
	for i in range(len(hid_all)):
		cnt += 1
		if verbose and cnt % 10 == 0:
			print(cnt / len(hid_all))
		if has_bh_cen[i] == 1:
			cen_bh_id[i] = bhid[(hid_bh == hid_all[i])]
			cen_bh_mass[i] = mbh[(hid_bh == hid_all[i])]
			cen_bh_mdot[i] = mdot[(hid_bh == hid_all[i])]
			cen_bh_mdot_mean[i] = mdot_mean[(hid_bh == hid_all[i])]
		if has_bh_any[i] == 1:
			any_bh_id[i] = bhid_any[(hid_bh_any == hid_all[i])]
			any_bh_mass[i] = mbh_any[(hid_bh_any == hid_all[i])]
			any_bh_mdot[i] = mdot_any[(hid_bh_any == hid_all[i])]
			any_bh_mdot_mean[i] = mdot_mean_any[(hid_bh_any == hid_all[i])]
		if calc_formation_times and mstar[i] * star_corr > t50min and mstar[i] * star_corr < t50max:
			target_halo = db.get_halo(step.path + '/' + str(hid_all[i]))
			mvir_time, mstar_time, time, redshift = target_halo.calculate_for_progenitors('Mvir', 'Mstar', 't()', 'z()')
			tmax_s = time[np.argmax(mstar_time)]
			tmax_mv = time[np.argmax(mvir_time)]
			tuse_mv = np.where(time < tmax_mv)[0]
			tuse_s = np.where(time < tmax_s)[0]
			if len(tuse_s) == 0 or len(tuse_mv) == 0:
				continue
			if mvir_time[tuse_mv].min() / mvir_time.max() > 0.5 or mstar_time[tuse_s].min() / mstar_time.max() > 0.5:
				continue
			else:
				mstar_max[i] = mstar_time.max()
				mvir_max[i] = mvir_time.max()
				ind_t80star = tuse_s[np.argmin(np.abs(mstar_time[tuse_s] - 0.8 * mstar_time.max()))]
				ind_t50star = tuse_s[np.argmin(np.abs(mstar_time[tuse_s] - 0.5 * mstar_time.max()))]

				ind_t80mv = tuse_mv[np.argmin(np.abs(mvir_time[tuse_mv] - 0.8 * mvir_time.max()))]
				ind_t50mv = tuse_mv[np.argmin(np.abs(mvir_time[tuse_mv] - 0.5 * mvir_time.max()))]

				t50_star_trace[i] = time[ind_t50star]
				t80_star_trace[i] = time[ind_t80star]

				t50_mvir_trace[i] = time[ind_t50mv]
				t80_mvir_trace[i] = time[ind_t80mv]

	data = {
		'mstar': mstar,
		'mvir': mvir,
		'hid': hid_all,
		'cen_bh_id': cen_bh_id,
		'any_bh_id': any_bh_id,
		'has_cen_bh': has_bh_cen,
		'has_any_bh': has_bh_any,
		'mbh_any': any_bh_mass,
		'mdot_any': any_bh_mdot,
		'mdot_mean_any': any_bh_mdot_mean,
		'mbh_cen': cen_bh_mass,
		'mdot_cen': cen_bh_mdot,
		'mdot_mean_cen': cen_bh_mdot_mean,
		'D': D
	}

	if calc_formation_times:
		data['t50_star'] = t50
		data['t80_star'] = t80
		data['t50_mvir_trace'] = t50_mvir_trace
		data['t80_mvir_trace'] = t80_mvir_trace
		data['t50_star_trace'] = t50_star_trace
		data['t80_star_trace'] = t80_star_trace
		data['mvir_max'] = mvir_max
		data['mstar_max'] = mstar_max

	data['notes'] = 'calculations for all halos with Mvir > 3e8 Msun (some are stripped so want to be less strict). Only include halos within 2 Mpc of cluster center'

	return data

def get_galaxy_bh_data_field(step, calc_formation_times=False, star_corr=0.6, verbose=True, t50min=1e7, t50max=1e9):
	print("getting database data for step", step)
	if calc_formation_times:
		hid_all, mstar, mvir, cen, r200, t50, t80 = step.gather_property('halo_number()', 'Mstar', 'Mvir', 'shrink_center',
	                                                                 'radius(200)', 'later(3).t50()', 'later(3).t80()')
	else:
		hid_all, mstar, mvir, cen, r200 = step.gather_property('halo_number()', 'Mstar', 'Mvir',
		                                                                 'shrink_center',
		                                                                 'radius(200)')
	hid_bh, mstar_bh, mvir_bh, mbh, mdot, mdot_mean, bhid = step.gather_property('halo_number()', 'Mstar',
	                                                                                           'Mvir',
	                                                                                           'link(BH_central, BH_mdot, "max", BH_central_distance<1).BH_mass',
	                                                                                           'link(BH_central, BH_mdot, "max", BH_central_distance<1).BH_mdot',
	                                                                                           'link(BH_central, BH_mdot, "max", BH_central_distance<1).BH_mdot_ave',
	                                                                                           'link(BH_central, BH_mdot, "max", BH_central_distance<1).halo_number()')

	hid_bh_any, mstar_bh_any, mvir_bh_any, mbh_any, mdot_any, mdot_mean_any, bhid_any= step.gather_property(
		'halo_number()',
		'Mstar', 'Mvir',
		'link(BH_central, BH_mdot, "max").BH_mass',
		'link(BH_central, BH_mdot, "max").BH_mdot',
		'link(BH_central, BH_mdot, "max").BH_mdot_ave',
		'link(BH_central, BH_mdot, "max").halo_number()')

	use = np.where(mvir > 3e9)[0]
	hid_all = hid_all[use]
	mstar = mstar[use]
	mvir = mvir[use]
	cen = cen[use]
	r200 = r200[use]

	# cross check with resolved halos
	use_bh_any = np.where(np.in1d(hid_bh_any, hid_all))[0]
	hid_bh_any = hid_bh_any[use_bh_any]
	mstar_bh_any = mstar_bh_any[use_bh_any]
	mbh_any = mbh_any[use_bh_any]
	mdot_any = mdot_any[use_bh_any]
	mdot_mean_any = mdot_mean_any[use_bh_any]
	bhid_any = bhid_any[use_bh_any]

	# cross check with resolved halos
	use_bh = np.where(np.in1d(hid_bh, hid_all))[0]
	bhid = bhid[use_bh]
	hid_bh = hid_bh[use_bh]
	mstar_bh = mstar_bh[use_bh]
	mvir_bh = mvir_bh[use_bh]
	mbh = mbh[use_bh]
	mdot = mdot[use_bh]
	mdot_mean = mdot_mean[use_bh]

	sat_flag_all = identify_sats_geha(cen, r200, mstar * star_corr, mvir, hid_all, scale=1 / (step.redshift + 1))

	has_bh_cen = np.zeros(len(hid_all))
	has_bh_any = np.zeros(len(hid_all))
	has_bh_cen[(np.in1d(hid_all, hid_bh))] = 1
	has_bh_any[(np.in1d(hid_all, hid_bh_any))] = 1

	cen_bh_id = np.ones(len(hid_all))*-1
	any_bh_id = np.ones(len(hid_all))*-1
	cen_bh_mass = np.ones(len(hid_all))*-1
	any_bh_mass = np.ones(len(hid_all))*-1
	cen_bh_mdot = np.ones(len(hid_all))*-1
	any_bh_mdot = np.ones(len(hid_all))*-1
	cen_bh_mdot_mean = np.ones(len(hid_all))*-1
	any_bh_mdot_mean = np.ones(len(hid_all))*-1

	if calc_formation_times:
		t50_star_trace = np.ones(len(hid_all))*-1
		t80_star_trace = np.ones(len(hid_all)) * -1
		t50_mvir_trace = np.ones(len(hid_all)) * -1
		t80_mvir_trace = np.ones(len(hid_all)) * -1
		mvir_max = np.ones(len(hid_all)) * -1
		mstar_max = np.ones(len(hid_all)) * -1
		print("collecting time evolution data...")

	cnt = 0
	print("collecting black hole data...")
	for i in range(len(hid_all)):
		cnt += 1
		if verbose and cnt%10==0:
			print(cnt/len(hid_all))
		if has_bh_cen[i]==1:
			cen_bh_id[i] = bhid[(hid_bh==hid_all[i])]
			cen_bh_mass[i] = mbh[(hid_bh==hid_all[i])]
			cen_bh_mdot[i] = mdot[(hid_bh==hid_all[i])]
			cen_bh_mdot_mean[i] = mdot_mean[(hid_bh==hid_all[i])]
		if has_bh_any[i]==1:
			any_bh_id[i] = bhid_any[(hid_bh_any == hid_all[i])]
			any_bh_mass[i] = mbh_any[(hid_bh_any == hid_all[i])]
			any_bh_mdot[i] = mdot_any[(hid_bh_any == hid_all[i])]
			any_bh_mdot_mean[i] = mdot_mean_any[(hid_bh_any == hid_all[i])]
		if calc_formation_times and mstar[i]*star_corr > t50min and mstar[i]*star_corr < t50max and sat_flag_all[i]==0:
			target_halo = db.get_halo(step.path+'/'+str(hid_all[i]))
			mvir_time, mstar_time, time, redshift = target_halo.calculate_for_progenitors('Mvir', 'Mstar', 't()', 'z()')
			tmax_s = time[np.argmax(mstar_time)]
			tmax_mv = time[np.argmax(mvir_time)]
			tuse_mv = np.where(time < tmax_mv)[0]
			tuse_s = np.where(time<tmax_s)[0]
			if len(tuse_s)==0 or len(tuse_mv)==0:
				continue
			if mvir_time[tuse_mv].min()/mvir_time.max() > 0.5 or mstar_time[tuse_s].min()/mstar_time.max() > 0.5:
				continue
			else:
				mstar_max[i] = mstar_time.max()
				mvir_max[i] = mvir_time.max()
				ind_t80star = tuse_s[np.argmin(np.abs(mstar_time[tuse_s]- 0.8*mstar_time.max()))]
				ind_t50star = tuse_s[np.argmin(np.abs(mstar_time[tuse_s] - 0.5 * mstar_time.max()))]

				ind_t80mv = tuse_mv[np.argmin(np.abs(mvir_time[tuse_mv] - 0.8 * mvir_time.max()))]
				ind_t50mv = tuse_mv[np.argmin(np.abs(mvir_time[tuse_mv] - 0.5 * mvir_time.max()))]

				t50_star_trace[i] = time[ind_t50star]
				t80_star_trace[i] = time[ind_t80star]

				t50_mvir_trace[i] = time[ind_t50mv]
				t80_mvir_trace[i] = time[ind_t80mv]

	data = {
		'mstar':mstar,
		'mvir': mvir,
		'hid': hid_all,
		'cen_bh_id': cen_bh_id,
		'any_bh_id': any_bh_id,
		'has_cen_bh': has_bh_cen,
		'has_any_bh': has_bh_any,
		'mbh_any': any_bh_mass,
		'mdot_any': any_bh_mdot,
		'mdot_mean_any': any_bh_mdot_mean,
		'mbh_cen': cen_bh_mass,
		'mdot_cen': cen_bh_mdot,
		'mdot_mean_cen': cen_bh_mdot_mean,
		'sat_flag': sat_flag_all
	}

	if calc_formation_times:
		data['t50_star'] = t50
		data['t80_star'] = t80
		data['t50_mvir_trace'] = t50_mvir_trace
		data['t80_mvir_trace'] = t80_mvir_trace
		data['t50_star_trace'] = t50_star_trace
		data['t80_star_trace'] = t80_star_trace
		data['mvir_max'] = mvir_max
		data['mstar_max'] = mstar_max

	data['notes'] = 'calculations for all halos with Mvir > 3e9 Msun. Isolation based on Geha+ 2012 definition'

	return data

def calculate_raw_occupation_fraction(timestep_data, mstar_bins = None, star_corr = 0.6, iso_only=False, central_only=False):
	if mstar_bins is None:
		mstar_bins = default_bins
	occ_frac = np.zeros(len(mstar_bins)-1)
	occ_frac_err = np.zeros((len(mstar_bins)-1,2))
	for i in range(len(mstar_bins) - 1):
		obin_all = np.where((np.log10(timestep_data['mstar'] * star_corr) > mstar_bins[i]) &
			        (np.log10(timestep_data['mstar'] * star_corr) < mstar_bins[i + 1]))[0]
		if len(obin_all)==0:
			occ_frac[i] = np.nan
			continue

		if iso_only:
			obin_all = obin_all[(timestep_data['sat_flag'][obin_all]==0)]
		if len(obin_all)==0:
			occ_frac[i] = np.nan
			continue

		if central_only:
			obin_bh = obin_all[(timestep_data['has_cen_bh'][obin_all]==1)]
		else:
			obin_bh = obin_all[(timestep_data['has_any_bh'][obin_all]== 1)]

		occ_frac_err[i] = binomial_errors(0.95, len(obin_bh), len(obin_all))
		occ_frac[i] = np.float(len(obin_bh))/len(obin_all)

	data = {
		'of': occ_frac,
		'of_err': occ_frac_err,
		'mstar_bins': mstar_bins,
		'notes': "Raw occupation fractions"
	}

	if iso_only:
		data['notes'] += " with only isolated halos"
	if central_only:
		data['notes'] += " and central D < 1 kpc BHs"

	return data

def calculate_cluster_distance_occupation_fraction(timestep_data,mstar_bins = None, star_corr = 0.6, central_only=False, distrange=[0,1], rvir=1036.5):
	if mstar_bins is None:
		mstar_bins = default_bins
	occ_frac = np.zeros(len(mstar_bins)-1)
	occ_frac_err = np.zeros((len(mstar_bins)-1,2))

	if 'D' not in timestep_data.keys():
		raise ValueError("distance information is not available in provided timestep data!")

	for i in range(len(mstar_bins) - 1):
		obin_all = np.where((np.log10(timestep_data['mstar'] * star_corr) > mstar_bins[i]) &
			        (np.log10(timestep_data['mstar'] * star_corr) < mstar_bins[i + 1]) &
			        (timestep_data['D']/rvir>distrange[0])&(timestep_data["D"]/rvir<distrange[1]))[0]
		if len(obin_all)==0:
			occ_frac[i] = np.nan
			continue

		if len(obin_all)==0:
			occ_frac[i] = np.nan
			continue

		if central_only:
			obin_bh = obin_all[(timestep_data['has_cen_bh'][obin_all]==1)]
		else:
			obin_bh = obin_all[(timestep_data['has_any_bh'][obin_all]== 1)]

		occ_frac_err[i] = binomial_errors(0.95, len(obin_bh), len(obin_all))
		occ_frac[i] = np.float(len(obin_bh))/len(obin_all)

	data = {
		'of': occ_frac,
		'of_err': occ_frac_err,
		'mstar_bins': mstar_bins,
		'notes': "Raw occupation fractions"
	}

	data['notes']+= " for halos at distances between "+str(distrange[0])+" and "+str(distrange[1])+\
	                " x Rvir ("+str(rvir)+" Mpc) from cluster center"
	if central_only:
		data['notes'] += " and only including central (D < 1 kpc) BHs"

	return data

def calculate_environment_occupation_Fraction(timestep_data, env_type='host', hostmassrange=[1e12,1e13], mstar_bins = None, star_corr = 0.6, central_only=False):
	host_mass_key = 'mvir_'+env_type
	if host_mass_key not in timestep_data.keys():
		raise ValueError("host information is not available in the provided timestep data!")
	if mstar_bins is None:
		mstar_bins = default_bins
	occ_frac = np.zeros(len(mstar_bins)-1)
	occ_frac_err = np.zeros((len(mstar_bins)-1,2))
	for i in range(len(mstar_bins) - 1):
		obin_all = np.where((np.log10(timestep_data['mstar'] * star_corr) > mstar_bins[i]) &
			        (np.log10(timestep_data['mstar'] * star_corr) < mstar_bins[i + 1]) &
			        (timestep_data[host_mass_key]>hostmassrange[0])&(timestep_data[host_mass_key]<hostmassrange[1]))[0]
		if len(obin_all)==0:
			occ_frac[i] = np.nan
			continue

		if len(obin_all)==0:
			occ_frac[i] = np.nan
			continue

		if central_only:
			obin_bh = obin_all[(timestep_data['has_cen_bh'][obin_all]==1)]
		else:
			obin_bh = obin_all[(timestep_data['has_any_bh'][obin_all]== 1)]

		occ_frac_err[i] = binomial_errors(0.95, len(obin_bh), len(obin_all))
		occ_frac[i] = np.float(len(obin_bh))/len(obin_all)

	data = {
		'of': occ_frac,
		'of_err': occ_frac_err,
		'mstar_bins': mstar_bins,
		'notes': "Raw occupation fractions"
	}

	if env_type=='host':
		data['notes'] += " for halos within 2 Rvir"
	if env_type=='close':
		data['notes'] += " for halos within 2 Mpc"
	data['notes']+= " of halos between "+str(hostmassrange[0])+" and "+str(hostmassrange[1])+" Msun"
	if central_only:
		data['notes'] += " and only including central (D < 1 kpc) BHs"

	return data

def calculate_luminous_occupation_Fraction(timestep_data, lx_cut, macc_dat = None, mstar_bins = None, star_corr = 0.6, iso_only=False,
                                           central_only=False, twomode=False, usemean=False):
	if mstar_bins is None:
		mstar_bins = default_bins

	mdot_key = 'mdot'
	mbh_key = 'mbh'
	if usemean:
		mdot_key += '_mean'
	if central_only:
		mbh_key += '_cen'
		mdot_key += '_cen'
		id_key = 'cen_bh_id'
		flag_key = 'has_cen_bh'
	else:
		mdot_key += '_any'
		mbh_key += '_any'
		id_key = 'any_bh_id'
		flag_key = 'has_any_bh'

	print('Searching BH data with the following keys based on inpuyt:', mdot_key, id_key, flag_key, mbh_key)

	bh_mdot_array = timestep_data[mdot_key]
	bh_mass_array = timestep_data[mbh_key]
	bh_id_array = timestep_data[id_key]
	lum_array = np.ones(len(bh_id_array))*-1

	mdot_units = pynbody.array.SimArray(bh_mdot_array, 'Msol yr**-1')
	if twomode:
		if macc_dat is None:
			raise RuntimeError("data for accreted mass not given!")
		for i in range(len(bh_id_array)):
			if timestep_data[flag_key][i]==0:
				continue
			if bh_id_array[i] not in macc_dat['bhid']:
				raise RuntimeError("BH id ", bh_id_array[i], "is not found in accreted mass data given")
			mbhacc = macc_dat['macc'][(macc_dat['bhid'] == bh_id_array[i])]
			#calculate based on Shen+ 2020 bolometric correction + two mode Lbol based on fedd (Sharma+ 2022)
			lum_array[i] = calc_xray_lum(pynbody.array.SimArray(mdot_units[i],mdot_units.units), mbhacc)
	else:
		lbol = 0.1 * Simpy.util.c.in_units('cm s**-1') ** 2 * \
		                                          mdot_units.in_units('g s**-1')[i]
		#apply Shen+ 2020 bolometric correction
		corr = calc_bol_corr_xray(lbol,band='soft') + calc_bol_corr_xray(lbol,band='hard')
		lum_array = lbol*corr

	occ_frac = np.zeros(len(mstar_bins) - 1)
	occ_frac_err = np.zeros((len(mstar_bins) - 1,2))
	for i in range(len(mstar_bins) - 1):
		obin_all = np.where((np.log10(timestep_data['mstar'] * star_corr) > mstar_bins[i]) &
		                    (np.log10(timestep_data['mstar'] * star_corr) < mstar_bins[i + 1]))[0]
		if len(obin_all) == 0:
			occ_frac[i] = np.nan
			continue

		if iso_only:
			obin_all = obin_all[(timestep_data['sat_flag'][obin_all] == 0)]
		if len(obin_all) == 0:
			occ_frac[i] = np.nan
			continue

		if central_only:
			obin_bh = obin_all[(timestep_data['has_cen_bh'][obin_all] == 1)&(lum_array[obin_all]>lx_cut)]
		else:
			obin_bh = obin_all[(timestep_data['has_any_bh'][obin_all] == 1)&(lum_array[obin_all]>lx_cut)]

		occ_frac_err[i] = binomial_errors(0.95, len(obin_bh), len(obin_all))
		occ_frac[i] = np.float(len(obin_bh)) / len(obin_all)

	data = {
		'of': occ_frac,
		'of_err': occ_frac_err,
		'mstar_bins': mstar_bins,
		'notes': "Raw occupation fractions with x-ray luminosity threshold "+ str(lx_cut) + ". Lx calculated with Shen+ 2020 bolometric correction."
	}
	if twomode:
		data['notes'] += " Two mode emission (sharma+ 2022) used to calculate Lbol."
	if iso_only:
		data['notes'] += " Only included isolated halos."
	if central_only:
		data['notes'] += " Only included central (D < 1 kpc) BHs."


	return data

def calculate_occupation_fraction_formation_time(timestep_data, mstar_range = None, formtimekey='t50_mstar_trace',
                                                 star_corr=0.6, central_only=False, tform_bins=None, iso_only=False, nmin=5):
	if mstar_range is None:
		mstar_range = [np.log10(np.min(timestep_data['mstar']*0.6)),np.log10(np.max(timestep_data['mstar']*0.6))]
	if tform_bins is None:
		tform_bins = default_time_bins
	if formtimekey not in timestep_data.keys():
		raise RuntimeError("provided formation time key", formtimekey, "not included in data!")
	if iso_only and 'sat_flag' not in timestep_data.keys():
		raise RuntimeError("sat flag is not provided in given dataset so cannot select isolated halos!")
	occ_frac = np.zeros(len(tform_bins)-1)
	occ_frac_err = np.zeros((len(tform_bins)-1,2))
	for i in range(len(tform_bins) - 1):
		obin_all = np.where((np.log10(timestep_data['mstar'] * star_corr) > mstar_range[0]) &
			        (np.log10(timestep_data['mstar'] * star_corr) < mstar_range[1]) &
			                (timestep_data[formtimekey]>tform_bins[i])&(timestep_data[formtimekey]<tform_bins[i+1]))[0]
		if len(obin_all)<nmin:
			occ_frac[i] = np.nan
			continue

		if iso_only:
			obin_all = obin_all[(timestep_data['sat_flag'][obin_all]==0)]
		if len(obin_all)==0:
			occ_frac[i] = np.nan
			continue

		if central_only:
			obin_bh = obin_all[(timestep_data['has_cen_bh'][obin_all]==1)]
		else:
			obin_bh = obin_all[(timestep_data['has_any_bh'][obin_all]== 1)]

		occ_frac_err[i] = binomial_errors(0.95, len(obin_bh), len(obin_all))
		occ_frac[i] = np.float(len(obin_bh))/len(obin_all)

	data = {
		'of': occ_frac,
		'of_err': occ_frac_err,
		'time_bins': tform_bins,
		'notes': "Raw occupation fractions vs formation time ("+formtimekey+")"
	}

	data['notes'] += " for galaxies with Mstar between "+str(mstar_range[0])+" and "+str(mstar_range[1])
	data['notes'] += " (after correction of "+str(star_corr)+")"

	if iso_only:
		data['notes'] += " with only isolated halos"
	if central_only:
		data['notes'] += " and central D < 1 kpc BHs"

	return data

def calculate_occupation_fraction_matched(timestep_data, match_data, mstar_bins=None, star_corr = 0.6,
                                          central_only=False, match_lim=4, matchall=True):
	if not mstar_bins:
		mstar_bins = default_bins
	occ_frac_match = np.zeros(len(mstar_bins) - 1)
	occ_frac_match_err = np.zeros((len(mstar_bins) - 1, 2))
	occ_frac = np.zeros(len(mstar_bins) - 1)
	occ_frac_err = np.zeros((len(mstar_bins) - 1, 2))

	if matchall:
		key_suffix='_all'
	else:
		key_suffix='_mvms'

	if central_only:
		bhkey = 'has_cen_bh'
	else:
		bhkey = 'has_any_bh'

	new_mstar_all = np.ones(len(timestep_data['hid']))*-1

	print("getting new Mstar values from match file...")
	for i in range(len(timestep_data['hid'])):
		curhid = timestep_data['hid'][i]
		ind_match = np.where((match_data['hid']==curhid)&(match_data['nmatch'+key_suffix]>=match_lim))[0]
		if len(ind_match)>1:
			raise RuntimeError("whoa something is weird here!!!!")
		if len(ind_match)==1:
			ind_match = ind_match[0]
			new_mstar_all[i] = match_data['Mstar'+key_suffix][ind_match]
		else:
			continue
	print(new_mstar_all)

	for i in range(len(mstar_bins) - 1):
		obin_all_match = np.where((new_mstar_all * star_corr > 10**mstar_bins[i]) &
			        (new_mstar_all * star_corr < 10**mstar_bins[i + 1]))[0]
		obin_all = np.where((new_mstar_all>0)&(timestep_data['mstar']*star_corr > 10**mstar_bins[i])&
		                    (timestep_data['mstar']*star_corr<10**mstar_bins[i+1]))[0]
		if len(obin_all)==0:
			occ_frac[i] = np.nan
		if len(obin_all_match)==0:
			occ_frac_match[i] = np.nan

		if len(obin_all)==0:
			occ_frac[i] = np.nan
			continue



		if len(obin_all)>0:
			obin_bh = obin_all[(timestep_data[bhkey][obin_all]==1)]
			occ_frac[i] = np.float(len(obin_bh)) / len(obin_all)
			occ_frac_err[i] = binomial_errors(0.95, len(obin_bh), len(obin_all))
		if len(obin_all_match)>0:
			obin_bh_match = obin_all_match[(timestep_data[bhkey][obin_all_match]==1)]
			occ_frac_match_err[i] = binomial_errors(0.95, len(obin_bh_match), len(obin_all_match))
			occ_frac_match[i] = np.float(len(obin_bh_match)) / len(obin_all_match)

	data = {
		'of': occ_frac,
		'of_err': occ_frac_err,
		'of_match': occ_frac_match,
		'of_match_err': occ_frac_match_err,
		'mstar_bins': mstar_bins,
		'notes': "Raw occupation fractions"
		}

	data['notes']+= " with stellar masses matched with"
	if matchall:
		data['notes']+= " Mstar, Mvir, and concentration"
	else:
		data['notes']+= " Mstar and Mvir"

	if central_only:
		data['notes'] += " and central D < 1 kpc BHs"

	return data

def get_cluster_occ_fractions(step, Lum_cut, mstar_bins = None, star_corr = 0.6, twomode=True, macc=None):
	if not mstar_bins:
		mstar_bins = default_bins
	print("loading in data for step", step)
	hid_all, mstar, mvir, cen, contam = step.gather_property('halo_number()', 'Mstar', 'Mvir',
										'shrink_center', 'contamination_fraction')
	hid_bh, mstar_bh, mvir_bh, mbh, mdot, mdot_mean, bhid = step.gather_property('halo_number()', 'Mstar', 'Mvir',
										'link(BH_central, BH_mdot, "max", BH_central_distance<1).BH_mass',
										'link(BH_central, BH_mdot, "max", BH_central_distance<1).BH_mdot',
										'link(BH_central, BH_mdot, "max", BH_central_distance<1).BH_mdot_ave',
	                                    'link(BH_central, BH_mdot, "max", BH_central_distance<1).halo_number()')

	hid_bh_any, mstar_bh_any, mvir_bh_any, mbh_any, mdot_any, mdot_mean_any = step.gather_property('halo_number()', 'Mstar', 'Mvir',
										'link(BH_central, BH_mdot, "max").BH_mass',
										'link(BH_central, BH_mdot, "max").BH_mdot',
										'link(BH_central, BH_mdot, "max").BH_mdot_ave')

	print("selecting resolved halos, crossmatching halos with black holes")

	darr = cen - cen[(hid_all==1)]
	Simpy.util.wrap(darr,boxsize=50e3,scale=1./(step.redshift+1))
	D = np.sqrt(np.sum(darr**2,axis=1))
	use = np.where((D<2000)&(contam<0.05)&(mvir>3e8))[0]
	hid_all = hid_all[use]
	mstar = mstar[use]
	mvir = mvir[use]

	use_bh = np.where(np.in1d(hid_bh,hid_all))[0]
	hid_bh = hid_bh[use_bh]
	mstar_bh = mstar_bh[use_bh]
	mvir_bh = mvir_bh[use_bh]
	bhid = bhid[use_bh]
	mbh = mbh[use_bh]
	mdot = mdot[use_bh]
	mdot_mean = mdot_mean[use_bh]

	#cross check with resolved halos
	use_bh_any = np.where(np.in1d(hid_bh_any,hid_all))[0]
	hid_bh_any = hid_bh_any[use_bh_any]
	mstar_bh_any = mstar_bh_any[use_bh_any]
	mvir_bh_any = mvir_bh_any[use_bh_any]
	mbh_any = mbh_any[use_bh_any]
	mdot_any = mdot_any[use_bh_any]
	mdot_mean_any = mdot_mean_any[use_bh_any]

	#Raw occupation fraction for ISOLATED galaxies
	print("Calculating raw occupation fractions for isolated galaxies")

	occ_frac = np.zeros(len(mstar_bins)-1)
	occ_frac_err = np.zeros((len(mstar_bins)-1, 2))

	for i in range(len(mstar_bins)-1):
		obin_all = np.where((np.log10(mstar*star_corr)>mstar_bins[i])&
				(np.log10(mstar*star_corr)<mstar_bins[i+1]))[0]
		obin_bh = np.where((np.log10(mstar_bh*star_corr)>mstar_bins[i])&
				(np.log10(mstar_bh*star_corr)<mstar_bins[i+1]))[0]

		if len(obin_all)>0:
			occ_frac_err[i] = binomial_errors(0.95, len(obin_bh), len(obin_all))
			occ_frac[i] = np.float(len(obin_bh))/len(obin_all)
		else:
			occ_frac[i] = np.nan

	#Occ fraction only for luminous, ISOLATED BHs
	print("calculating luminous occupation fractions")

	mdot_units = pynbody.array.SimArray(mdot, 'Msol yr**-1')
	mdot_mean_units = pynbody.array.SimArray(mdot_mean, 'Msol yr**-1')
	if twomode:
		if not macc:
			raise RuntimeError("data for accreted mass not given!")
		mbhacc = np.zeros(len(mbh))
		for i in range(len(mbh)):
			if bhid[i] not in macc['bhid']:
				raise RuntimeError("BH id ", bhid[i], "is not found in accreted mass data given")
			mbhacc[i] = macc['macc'][(macc['bhid']==bhid[i])]
		lum = calc_xray_lum(mdot_units, mbhacc)
		lum_2 = calc_xray_lum(mdot_mean_units,mbhacc)
	else:
		lum = 0.1*Simpy.util.c.in_units('cm s**-1')**2*mdot_units.in_units('g s**-1')
		lum_2 = 0.1*Simpy.util.c.in_units('cm s**-1')**2*mdot_mean_units.in_units('g s**-1')
	occ_frac_lum = np.zeros(len(mstar_bins)-1)
	occ_frac_lum_2 = np.zeros(len(mstar_bins)-1)
	occ_frac_lum_err = np.zeros((len(mstar_bins)-1, 2))
	occ_frac_lum_2_err = np.zeros((len(mstar_bins)-1, 2))

	for i in range(len(mstar_bins)-1):
		obin_all = np.where((np.log10(mstar*star_corr)>mstar_bins[i])&
					(np.log10(mstar*star_corr)<mstar_bins[i+1]))[0]
		obin_bh = np.where((np.log10(mstar_bh*star_corr)>mstar_bins[i])&
					(np.log10(mstar_bh*star_corr)<mstar_bins[i+1])&
					(lum>Lum_cut))[0]
		obin_bh_2 = np.where((np.log10(mstar_bh*star_corr)>mstar_bins[i])&
					(np.log10(mstar_bh*star_corr)<mstar_bins[i+1])&
					(lum_2>Lum_cut))[0]
		if len(obin_all)>0:
			occ_frac_lum[i] = np.float(len(obin_bh))/len(obin_all)
			occ_frac_lum_2[i] = np.float(len(obin_bh_2))/len(obin_all)
			occ_frac_lum_err[i] = binomial_errors(0.95, len(obin_bh), len(obin_all))
			occ_frac_lum_2_err[i] = binomial_errors(0.95, len(obin_bh_2), len(obin_all))
		else:
			occ_frac_lum[i] = np.nan
			occ_frac_lum_2[i] = np.nan

	#Occ fraction only for any BHs
	print("calculating occupation fractionsn for all BHs, including wanderers")

	occ_frac_any = np.zeros(len(mstar_bins)-1)
	occ_frac_any_err = np.zeros((len(mstar_bins)-1, 2))

	for i in range(len(mstar_bins)-1):
		obin_all = np.where((np.log10(mstar*star_corr)>mstar_bins[i])&
				(np.log10(mstar*star_corr)<mstar_bins[i+1]))[0]
		obin_bh = np.where((np.log10(mstar_bh_any*star_corr)>mstar_bins[i])&
				(np.log10(mstar_bh_any*star_corr)<mstar_bins[i+1]))[0]

		if len(obin_all)>0:
			occ_frac_any_err[i] = binomial_errors(0.95, len(obin_bh), len(obin_all))
			occ_frac_any[i] = np.float(len(obin_bh))/len(obin_all)

		else:
			occ_frac_any[i] = np.nan


	data = {'mstar_bins':mstar_bins,
			'of_all': occ_frac,
			'of_all_err':occ_frac_err,
			'of_lum': occ_frac_lum,
			'of_lum_err':occ_frac_lum_err,
			'of_lum_1.6Myr':occ_frac_lum_2,
			'of_lum_1.6Myr_err':occ_frac_lum_2_err,
			'of_any': occ_frac_any,
			'of_any_err':occ_frac_any_err,
			'notes':'lum cut used: %1.3E' % Lum_cut}

	return data

def get_field_occ_fractions(step, Lum_cut, mstar_bins=None,star_corr=0.6, twomode=True, macc=None):
	if not mstar_bins:
		mstar_bins = default_bins
	print("loading in data for step", step)
	hid_all, mstar, mvir, cen, r200 = step.gather_property('halo_number()', 'Mstar', 'Mvir', 'shrink_center', 'radius(200)')
	hid_bh, mstar_bh, mvir_bh, mbh, mdot, mdot_mean, bhid = step.gather_property('halo_number()', 'Mstar', 'Mvir',
													'link(BH_central, BH_mdot, "max", BH_central_distance<1).BH_mass',
													'link(BH_central, BH_mdot, "max", BH_central_distance<1).BH_mdot',
													'link(BH_central, BH_mdot, "max", BH_central_distance<1).BH_mdot_ave',
													'link(BH_central, BH_mdot, "max", BH_central_distance<1).halo_number()')

	hid_bh_any, mstar_bh_any, mvir_bh_any, mbh_any, mdot_any, mdot_mean_any = step.gather_property('halo_number()',
													'Mstar', 'Mvir',
													'link(BH_central, BH_mdot, "max").BH_mass',
													'link(BH_central, BH_mdot, "max").BH_mdot',
													'link(BH_central, BH_mdot, "max").BH_mdot_ave')

	hid_sf, mstar_sf, r200_sf, sfr25, sfr250, sfr25_10, sfr250_10 = step.gather_property('halo_number()', 'Mstar',
																	'radius(200)','at(0.1*radius(200),SFR_encl_25Myr)',
																	'at(0.1*radius(200),SFR_encl_250Myr)',
																	'at(10,SFR_encl_250Myr)',
																	'at(10,SFR_encl_250Myr)')
	sfr = np.copy(sfr25)
	sfr[(sfr <= 0.0064)] = sfr250[(sfr < 0.0064)]
	sfr[(r200_sf < 100)] = sfr25_10[(r200_sf < 100)]
	sfr[(r200_sf < 100) & (sfr <= 0.0064)] = sfr250_10[(r200_sf < 100) & (sfr <= 0.0064)]

	qf = identify_quench(mstar_sf * 0.6, sfr, max(0.0, step.redshift), 1.0)
	hid_quench = hid_sf[(qf == 1)]

	print("selecting resolved halos, crossmatching halos with black holes")

	# Only select halos we consider resolved
	use = np.where(mvir > 3e9)[0]
	hid_all = hid_all[use]
	mstar = mstar[use]
	mvir = mvir[use]
	cen = cen[use]
	r200 = r200[use]

	# cross check with resolved halos
	use_bh_any = np.where(np.in1d(hid_bh_any, hid_all))[0]
	hid_bh_any = hid_bh_any[use_bh_any]
	mstar_bh_any = mstar_bh_any[use_bh_any]
	mvir_bh_any = mvir_bh_any[use_bh_any]
	mbh_any = mbh_any[use_bh_any]
	mdot_any = mdot_any[use_bh_any]
	mdot_mean_any = mdot_mean_any[use_bh_any]

	# cross check with resolved halos
	use_bh = np.where(np.in1d(hid_bh, hid_all))[0]
	hid_bh = hid_bh[use_bh]
	bhid = bhid[use_bh]
	mstar_bh = mstar_bh[use_bh]
	mvir_bh = mvir_bh[use_bh]
	mbh = mbh[use_bh]
	mdot = mdot[use_bh]
	mdot_mean = mdot_mean[use_bh]

	sat_flag_all = identify_sats_geha(cen, r200, mstar * star_corr, mvir, hid_all, scale=1 / (1+step.redshift))

	# Raw occupation fraction for ISOLATED galaxies
	print("Calculating raw occupation fractions for isolated galaxies")

	occ_frac = np.zeros(len(mstar_bins) - 1)
	occ_frac_err = np.zeros((len(mstar_bins)-1, 2))
	occ_frac_iso = np.zeros(len(mstar_bins) - 1)
	occ_frac_err_iso = np.zeros((len(mstar_bins)-1, 2))

	for i in range(len(mstar_bins) - 1):
		obin_all = np.where((np.log10(mstar * star_corr) > mstar_bins[i]) &
						(np.log10(mstar * star_corr) < mstar_bins[i + 1]))[0]
		obin_bh = np.where((np.log10(mstar_bh * star_corr) > mstar_bins[i]) &
						(np.log10(mstar_bh * star_corr) < mstar_bins[i + 1]))[0]

		obin_all_iso = np.where((np.log10(mstar * star_corr) > mstar_bins[i]) &
						(np.log10(mstar * star_corr) < mstar_bins[i + 1]) &
						(sat_flag_all == 0))[0]
		obin_bh_iso = np.where((np.log10(mstar_bh * star_corr) > mstar_bins[i]) &
						(np.log10(mstar_bh * star_corr) < mstar_bins[i + 1]) &
						(np.in1d(hid_bh, hid_all[(sat_flag_all == 0)])))[0]

		if len(obin_all) > 0:
			occ_frac_err[i] = binomial_errors(0.95, len(obin_bh), len(obin_all))
			occ_frac[i] = np.float(len(obin_bh)) / len(obin_all)
		else:
			occ_frac[i] = np.nan

		if len(obin_all_iso) > 0:
			occ_frac_iso[i] = np.float(len(obin_bh_iso)) / len(obin_all_iso)
			occ_frac_err_iso[i] = binomial_errors(0.95, len(obin_bh_iso), len(obin_all_iso))
		else:
			occ_frac_iso[i] = np.nan

	# Occ fraction only for luminous, ISOLATED BHs
	print("calculating luminous occupation fractions")

	mdot_units = pynbody.array.SimArray(mdot, 'Msol yr**-1')
	mdot_mean_units = pynbody.array.SimArray(mdot_mean, 'Msol yr**-1')
	if twomode:
		if not macc:
			raise RuntimeError("data for accreted mass not given!")
		mbhacc = np.zeros(len(mbh))
		for i in range(len(mbh)):
			if bhid[i] not in macc['bhid']:
				raise RuntimeError("BH id ", bhid[i], "is not found in accreted mass data given")
			mbhacc[i] = macc['macc'][(macc['bhid'] == bhid[i])]
		lum = calc_xray_lum(mdot_units, mbhacc)
		lum_2 = calc_xray_lum(mdot_mean_units, mbhacc)
	else:
		lum = 0.1 * Simpy.util.c.in_units('cm s**-1') ** 2 * mdot_units.in_units('g s**-1')
		lum_2 = 0.1 * Simpy.util.c.in_units('cm s**-1') ** 2 * mdot_mean_units.in_units('g s**-1')
	occ_frac_lum = np.zeros(len(mstar_bins) - 1)
	occ_frac_lum_2 = np.zeros(len(mstar_bins) - 1)
	occ_frac_lum_iso = np.zeros(len(mstar_bins) - 1)
	occ_frac_lum_2_iso = np.zeros(len(mstar_bins) - 1)
	occ_frac_lum_err = np.zeros((len(mstar_bins)-1, 2))
	occ_frac_lum_2_err = np.zeros((len(mstar_bins)-1, 2))
	occ_frac_lum_iso_err = np.zeros((len(mstar_bins)-1, 2))
	occ_frac_lum_2_iso_err = np.zeros((len(mstar_bins)-1, 2))

	occ_frac_lum_quench = np.zeros(len(mstar_bins) - 1)
	occ_frac_lum_2_quench = np.zeros(len(mstar_bins) - 1)
	occ_frac_lum_quench_err = np.zeros((len(mstar_bins)-1, 2))
	occ_frac_lum_2_quench_err = np.zeros((len(mstar_bins)-1, 2))

	for i in range(len(mstar_bins) - 1):
		obin_all = np.where((np.log10(mstar * star_corr) > mstar_bins[i]) &
								(np.log10(mstar * star_corr) < mstar_bins[i + 1]))[0]
		obin_bh = np.where((np.log10(mstar_bh * star_corr) > mstar_bins[i]) &
								(np.log10(mstar_bh * star_corr) < mstar_bins[i + 1]) &
		                   (lum > Lum_cut))[0]
		obin_bh_2 = np.where((np.log10(mstar_bh * star_corr) > mstar_bins[i]) &
								(np.log10(mstar_bh * star_corr) < mstar_bins[i + 1]) &
								(lum_2 > Lum_cut))[0]

		obin_all_iso = np.where((np.log10(mstar * star_corr) > mstar_bins[i]) &
								(np.log10(mstar * star_corr) < mstar_bins[i + 1]) & (sat_flag_all == 0))[0]
		obin_bh_iso = np.where((np.log10(mstar_bh * star_corr) > mstar_bins[i]) &
								(np.log10(mstar_bh * star_corr) < mstar_bins[i + 1]) &
								(lum > Lum_cut) & (np.in1d(hid_bh, hid_all[(sat_flag_all == 0)])))[0]
		obin_bh_2_iso = np.where((np.log10(mstar_bh * star_corr) > mstar_bins[i]) &
								(np.log10(mstar_bh * star_corr) < mstar_bins[i + 1]) &
								(lum_2 > Lum_cut) & (np.in1d(hid_bh, hid_all[(sat_flag_all == 0)])))[0]

		obin_quench = np.where((np.log10(mstar * star_corr) > mstar_bins[i]) &
								(np.log10(mstar * star_corr) < mstar_bins[i + 1]) & (np.in1d(hid_all, hid_quench)))[0]
		obin_bh_quench = np.where((np.log10(mstar_bh * star_corr) > mstar_bins[i]) &
								(np.log10(mstar_bh * star_corr) < mstar_bins[i + 1]) &
		                          (lum > Lum_cut) & (np.in1d(hid_bh, hid_quench)))[0]
		obin_bh_2_quench = np.where((np.log10(mstar_bh * star_corr) > mstar_bins[i]) &
								(np.log10(mstar_bh * star_corr) < mstar_bins[i + 1]) &
								(lum_2 > Lum_cut) & (np.in1d(hid_bh, hid_quench)))[0]

		if len(obin_all) > 0:
			occ_frac_lum[i] = np.float(len(obin_bh)) / len(obin_all)
			occ_frac_lum_2[i] = np.float(len(obin_bh_2)) / len(obin_all)
			occ_frac_lum_err[i] = binomial_errors(0.95, len(obin_bh), len(obin_all))
			occ_frac_lum_2_err[i] = binomial_errors(0.95, len(obin_bh_2), len(obin_all))
		else:
			occ_frac_lum[i] = np.nan
			occ_frac_lum_2[i] = np.nan

		if len(obin_all_iso) > 0:
			occ_frac_lum_iso[i] = np.float(len(obin_bh_iso)) / len(obin_all_iso)
			occ_frac_lum_2_iso[i] = np.float(len(obin_bh_2_iso)) / len(obin_all_iso)
			occ_frac_lum_iso_err[i] = binomial_errors(0.95, len(obin_bh_iso), len(obin_all_iso))
			occ_frac_lum_2_iso_err[i] = binomial_errors(0.95, len(obin_bh_2_iso), len(obin_all_iso))
		else:
			occ_frac_lum_iso[i] = np.nan
			occ_frac_lum_2_iso[i] = np.nan

		if len(obin_quench) > 0:
			occ_frac_lum_quench[i] = np.float(len(obin_bh_quench)) / len(obin_quench)
			occ_frac_lum_2_quench[i] = np.float(len(obin_bh_2_quench)) / len(obin_quench)
			occ_frac_lum_quench_err[i] = binomial_errors(0.95, len(obin_bh_quench), len(obin_quench))
			occ_frac_lum_2_quench_err[i] = binomial_errors(0.95, len(obin_bh_2_quench), len(obin_quench))
		else:
			occ_frac_lum_quench[i] = np.nan
			occ_frac_lum_2_quench[i] = np.nan

	# Occ fraction only for any BHs
	print("calculating occupation fractionsn for all BHs, including wanderers")

	occ_frac_any = np.zeros(len(mstar_bins) - 1)
	occ_frac_any_err = np.zeros((len(mstar_bins) - 1, 2))
	occ_frac_any_iso = np.zeros(len(mstar_bins) - 1)
	occ_frac_any_err_iso = np.zeros((len(mstar_bins) - 1, 2))

	for i in range(len(mstar_bins) - 1):
		obin_all = np.where((np.log10(mstar * star_corr) > mstar_bins[i]) &
								(np.log10(mstar * star_corr) < mstar_bins[i + 1]))[0]
		obin_bh = np.where((np.log10(mstar_bh_any * star_corr) > mstar_bins[i]) &
								(np.log10(mstar_bh_any * star_corr) < mstar_bins[i + 1]))[0]

		obin_all_iso = np.where((np.log10(mstar * star_corr) > mstar_bins[i]) &
								(np.log10(mstar * star_corr) < mstar_bins[i + 1]) &
								(sat_flag_all == 0))[0]
		obin_bh_iso = np.where((np.log10(mstar_bh_any * star_corr) > mstar_bins[i]) &
								(np.log10(mstar_bh_any * star_corr) < mstar_bins[i + 1]) &
								(np.in1d(hid_bh_any, hid_all[(sat_flag_all == 0)])))[0]

		if len(obin_all) > 0:
			occ_frac_any_err[i] = binomial_errors(0.95, len(obin_bh), len(obin_all))
			occ_frac_any[i] = np.float(len(obin_bh)) / len(obin_all)
		else:
			occ_frac_any[i] = np.nan

		if len(obin_all_iso) > 0:
			occ_frac_any_iso[i] = np.float(len(obin_bh_iso)) / len(obin_all_iso)
			occ_frac_any_err_iso[i] = binomial_errors(0.95, len(obin_bh_iso), len(obin_all_iso))
		else:
			occ_frac_any_iso[i] = np.nan

	data = {'mstar_bins': mstar_bins,
			'of_all': occ_frac,
			'of_all_err': occ_frac_err,
			'of_iso': occ_frac_iso,
			'of_iso_err': occ_frac_err_iso,
			'of_lum': occ_frac_lum,
			'of_lum_err': occ_frac_lum_err,
			'of_lum_1.6Myr': occ_frac_lum_2,
			'of_lum_1.6Myr_err': occ_frac_lum_2_err,
			'of_lum_iso': occ_frac_lum_iso,
			'of_lum_iso_err': occ_frac_lum_iso_err,
			'of_lum_1.6Myr_iso': occ_frac_lum_2_iso,
			'of_lum_1.6Myr_iso_err': occ_frac_lum_2_iso_err,
			'of_any': occ_frac_any,
			'of_any_err': occ_frac_any_err,
			'of_any_iso': occ_frac_any_iso,
			'of_any_err_iso': occ_frac_any_err_iso,
			'of_lum_quench': occ_frac_lum_quench,
			'of_lum_1.6Myr_quench': occ_frac_lum_2_quench,
			'of_lum_quench_err': occ_frac_lum_quench_err,
			'of_lum_1.6Myr_quench_err': occ_frac_lum_2_quench_err,
			'notes': 'lum cut used: %1.3E' % Lum_cut}

	return data

def get_field_occ_fractions_with_formation_time(step, mstar_bins=None,star_corr=0.6, time_bins=None):
	if not mstar_bins:
		mstar_bins = default_bins
	if not time_bins:
		time_bins = np.arange(0,15,1)
	print("loading in data for step", step)
	hid_all, mstar, mvir, cen, r200, t50, t80 = step.gather_property('halo_number()', 'Mstar', 'Mvir', 'shrink_center', 'radius(200)', 't50()','t80()')
	hid_bh, mstar_bh, mvir_bh, mbh, mdot, mdot_mean, bhid, t50bh, t80bh = step.gather_property('halo_number()', 'Mstar', 'Mvir',
													'link(BH_central, BH_mdot, "max", BH_central_distance<1).BH_mass',
													'link(BH_central, BH_mdot, "max", BH_central_distance<1).BH_mdot',
													'link(BH_central, BH_mdot, "max", BH_central_distance<1).BH_mdot_ave',
													'link(BH_central, BH_mdot, "max", BH_central_distance<1).halo_number()', 't50()','t80()')

	hid_bh_any, mstar_bh_any, mvir_bh_any, mbh_any, mdot_any, mdot_mean_any,t50bh_any, t80bh_any = step.gather_property('halo_number()',
													'Mstar', 'Mvir',
													'link(BH_central, BH_mdot, "max").BH_mass',
													'link(BH_central, BH_mdot, "max").BH_mdot',
													'link(BH_central, BH_mdot, "max").BH_mdot_ave','t50()','t80()')

	use = np.where(mvir > 3e9)[0]
	hid_all = hid_all[use]
	mstar = mstar[use]
	mvir = mvir[use]
	cen = cen[use]
	r200 = r200[use]

	# cross check with resolved halos
	use_bh_any = np.where(np.in1d(hid_bh_any, hid_all))[0]
	hid_bh_any = hid_bh_any[use_bh_any]
	mstar_bh_any = mstar_bh_any[use_bh_any]

	# cross check with resolved halos
	use_bh = np.where(np.in1d(hid_bh, hid_all))[0]
	hid_bh = hid_bh[use_bh]
	mstar_bh = mstar_bh[use_bh]
	mvir_bh = mvir_bh[use_bh]
	mbh = mbh[use_bh]
	mdot = mdot[use_bh]
	mdot_mean = mdot_mean[use_bh]

	sat_flag_all = identify_sats_geha(cen, r200, mstar * star_corr, mvir, hid_all, scale=1 / (step.redshfit+1))

	# Raw occupation fraction for ISOLATED galaxies
	print("Calculating raw occupation fractions for isolated galaxies")

	occ_frac = np.zeros((len(mstar_bins) - 1,len(time_bins)-1))
	occ_frac_err = np.zeros((len(mstar_bins) - 1, len(time_bins)-1, 2))
	occ_frac_iso = np.zeros((len(mstar_bins) - 1, len(time_bins)-1))
	occ_frac_err_iso = np.zeros((len(mstar_bins) - 1, len(time_bins)-1, 2))

	occ_frac_any = np.zeros((len(mstar_bins) - 1, len(time_bins) - 1))
	occ_frac_any_err = np.zeros((len(mstar_bins) - 1, len(time_bins) - 1, 2))
	occ_frac_any_iso = np.zeros((len(mstar_bins) - 1, len(time_bins) - 1))
	occ_frac_any_err_iso = np.zeros((len(mstar_bins) - 1, len(time_bins) - 1, 2))

	occ_frac_80 = np.zeros((len(mstar_bins) - 1, len(time_bins) - 1))
	occ_frac_80_err = np.zeros((len(mstar_bins) - 1, len(time_bins) - 1, 2))
	occ_frac_80_iso = np.zeros((len(mstar_bins) - 1, len(time_bins) - 1))
	occ_frac_80_err_iso = np.zeros((len(mstar_bins) - 1, len(time_bins) - 1, 2))

	occ_frac_80_any = np.zeros((len(mstar_bins) - 1, len(time_bins) - 1))
	occ_frac_80_any_err = np.zeros((len(mstar_bins) - 1, len(time_bins) - 1, 2))
	occ_frac_80_any_iso = np.zeros((len(mstar_bins) - 1, len(time_bins) - 1))
	occ_frac_80_any_err_iso = np.zeros((len(mstar_bins) - 1, len(time_bins) - 1, 2))


	for i in range(len(mstar_bins) - 1):
		for j in range(len(time_bins)-1):

			obin_all = np.where((np.log10(mstar * star_corr) > mstar_bins[i]) &
						(np.log10(mstar * star_corr) < mstar_bins[i + 1]) &
						(t50 > time_bins[j]) & (t50 < time_bins[j+1]))[0]
			obin_bh = np.where((np.log10(mstar_bh * star_corr) > mstar_bins[i]) &
						(np.log10(mstar_bh * star_corr) < mstar_bins[i + 1]) &
			            (t50bh > time_bins[j]) & (t50bh < time_bins[j+1]))[0]
			obin_bh_any = np.where((np.log10(mstar_bh_any * star_corr) > mstar_bins[i]) &
			                   (np.log10(mstar_bh_any * star_corr) < mstar_bins[i + 1]) &
			                   (t80bh_any > time_bins[j]) & (t80bh_any < time_bins[j + 1]))[0]

			obin_all_iso = np.where((np.log10(mstar * star_corr) > mstar_bins[i]) &
						(np.log10(mstar * star_corr) < mstar_bins[i + 1]) &
						(t50bh > time_bins[j]) & (t50bh < time_bins[j + 1]) &
						(sat_flag_all == 0))[0]
			obin_bh_iso = np.where((np.log10(mstar_bh * star_corr) > mstar_bins[i]) &
						(np.log10(mstar_bh * star_corr) < mstar_bins[i + 1]) &
						(t50bh > time_bins[j]) & (t50bh < time_bins[j + 1]) &
						(np.in1d(hid_bh, hid_all[(sat_flag_all == 0)])))[0]
			obin_bh_any_iso = np.where((np.log10(mstar_bh_any * star_corr) > mstar_bins[i]) &
			                       (np.log10(mstar_bh_any * star_corr) < mstar_bins[i + 1]) &
			                       (t50bh_any > time_bins[j]) & (t50bh_any < time_bins[j + 1]) &
			                       (np.in1d(hid_bh_any, hid_all[(sat_flag_all == 0)])))[0]

			if len(obin_all) > 0:
				occ_frac_err[i,j] = binomial_errors(0.95, len(obin_bh), len(obin_all))
				occ_frac[i,j] = np.float(len(obin_bh)) / len(obin_all)
				occ_frac_any[i,j] = np.float(len(obin_bh_any)) / len(obin_all)
				occ_frac_any_err[i, j] = binomial_errors(0.95, len(obin_bh_any), len(obin_all))
			else:
				occ_frac[i,j] = np.nan
				occ_frac_any[i,j] = np.nan

			if len(obin_all_iso) > 0:
				occ_frac_iso[i,j] = np.float(len(obin_bh_iso)) / len(obin_all_iso)
				occ_frac_err_iso[i,j] = binomial_errors(0.95, len(obin_bh_iso), len(obin_all_iso))
				occ_frac_any_iso[i, j] = np.float(len(obin_bh_any_iso)) / len(obin_all_iso)
				occ_frac_any_err_iso[i, j] = binomial_errors(0.95, len(obin_bh_any_iso), len(obin_all_iso))
			else:
				occ_frac_iso[i,j] = np.nan
				occ_frac_any_iso[i,j] = np.nan

		for i in range(len(mstar_bins) - 1):
			for j in range(len(time_bins) - 1):

				obin_all = np.where((np.log10(mstar * star_corr) > mstar_bins[i]) &
				                    (np.log10(mstar * star_corr) < mstar_bins[i + 1]) &
				                    (t80 > time_bins[j]) & (t80 < time_bins[j + 1]))[0]
				obin_bh = np.where((np.log10(mstar_bh * star_corr) > mstar_bins[i]) &
				                   (np.log10(mstar_bh * star_corr) < mstar_bins[i + 1]) &
				                   (t80bh > time_bins[j]) & (t80bh < time_bins[j + 1]))[0]
				obin_bh_any = np.where((np.log10(mstar_bh_any * star_corr) > mstar_bins[i]) &
				                       (np.log10(mstar_bh_any * star_corr) < mstar_bins[i + 1]) &
				                       (t80bh_any > time_bins[j]) & (t80bh_any < time_bins[j + 1]))[0]

				obin_all_iso = np.where((np.log10(mstar * star_corr) > mstar_bins[i]) &
				                        (np.log10(mstar * star_corr) < mstar_bins[i + 1]) &
				                        (t80bh > time_bins[j]) & (t80bh < time_bins[j + 1]) &
				                        (sat_flag_all == 0))[0]
				obin_bh_iso = np.where((np.log10(mstar_bh * star_corr) > mstar_bins[i]) &
				                       (np.log10(mstar_bh * star_corr) < mstar_bins[i + 1]) &
				                       (t80bh > time_bins[j]) & (t80bh < time_bins[j + 1]) &
				                       (np.in1d(hid_bh, hid_all[(sat_flag_all == 0)])))[0]
				obin_bh_any_iso = np.where((np.log10(mstar_bh_any * star_corr) > mstar_bins[i]) &
				                           (np.log10(mstar_bh_any * star_corr) < mstar_bins[i + 1]) &
				                           (t80bh_any > time_bins[j]) & (t80bh_any < time_bins[j + 1]) &
				                           (np.in1d(hid_bh_any, hid_all[(sat_flag_all == 0)])))[0]

				if len(obin_all) > 0:
					occ_frac_80_err[i, j] = binomial_errors(0.95, len(obin_bh), len(obin_all))
					occ_frac_80[i, j] = np.float(len(obin_bh)) / len(obin_all)
					occ_frac_80_any[i, j] = np.float(len(obin_bh_any)) / len(obin_all)
					occ_frac_80_any_err[i, j] = binomial_errors(0.95, len(obin_bh_any), len(obin_all))
				else:
					occ_frac_80[i, j] = np.nan
					occ_frac_80_any[i, j] = np.nan

				if len(obin_all_iso) > 0:
					occ_frac_80_iso[i, j] = np.float(len(obin_bh_iso)) / len(obin_all_iso)
					occ_frac_80_err_iso[i, j] = binomial_errors(0.95, len(obin_bh_iso), len(obin_all_iso))
					occ_frac_80_any_iso[i, j] = np.float(len(obin_bh_any_iso)) / len(obin_all_iso)
					occ_frac_80_any_err_iso[i, j] = binomial_errors(0.95, len(obin_bh_any_iso), len(obin_all_iso))
				else:
					occ_frac_80_iso[i, j] = np.nan
					occ_frac_80_any_iso[i, j] = np.nan

	data = {'mstar_bins': mstar_bins,
	        'time_bins': time_bins,
	        'of_all_t50': occ_frac,
	        'of_iso_t50': occ_frac_iso,
	        'of_any_t50': occ_frac_any,
	        'of_any_iso_t50': occ_frac_any_iso,
	        'of_all_t50_err': occ_frac_err,
	        'of_iso_t50_err': occ_frac_err_iso,
	        'of_any_t50_err': occ_frac_any_err,
	        'of_any_iso_t50_err': occ_frac_any_err_iso,
	        'of_all_t80': occ_frac_80,
	        'of_iso_t80': occ_frac_80_iso,
	        'of_any_t80': occ_frac_80_any,
	        'of_any_iso_t80': occ_frac_80_any_iso,
	        'of_all_t80_err': occ_frac_80_err,
	        'of_iso_t80_err': occ_frac_80_err_iso,
	        'of_any_t80_err': occ_frac_80_any_err,
	        'of_any_iso_t80_err': occ_frac_80_any_err_iso,
	        'notes': "occupation fraction calculated in bins of t50 and t80, for central (D<1) or any BH in any environment or isolated"
	        }

	return data

def get_matched_occupation_fractions(step, Lum_cut, matched, mstar_bins = None, star_corr = 0.6, use_con=True, match_limit=4):
	if not mstar_bins:
		mstar_bins = default_bins
	print("loading in data for step", step)

	hid_all, mstar, mvir, cen, contam = step.gather_property('halo_number()', 'Mstar', 'Mvir',
	                                                         'shrink_center', 'contamination_fraction')
	hid_bh, mstar_bh, mvir_bh, mbh, mdot, mdot_mean, bhid = step.gather_property('halo_number()', 'Mstar', 'Mvir',
	                                     'link(BH_central, BH_mdot, "max", BH_central_distance<1).BH_mass',
	                                    'link(BH_central, BH_mdot, "max", BH_central_distance<1).BH_mdot',
	                                     'link(BH_central, BH_mdot, "max", BH_central_distance<1).BH_mdot_ave',
	                                     'link(BH_central, BH_mdot, "max", BH_central_distance<1).halo_number()')

	hid_bh_any, mstar_bh_any, mvir_bh_any, mbh_any, mdot_any, mdot_mean_any = step.gather_property('halo_number()',
	                                    'Mstar', 'Mvir',
	                                    'link(BH_central, BH_mdot, "max").BH_mass',
	                                    'link(BH_central, BH_mdot, "max").BH_mdot',
	                                    'link(BH_central, BH_mdot, "max").BH_mdot_ave')

	darr = cen - cen[(hid_all == 1)]
	Simpy.util.wrap(darr, boxsize=50e3, scale=1. / (step.redshift + 1))
	D = np.sqrt(np.sum(darr ** 2, axis=1))
	use = np.where((D < 2000) & (contam < 0.05) & (mvir > 3e8) & (mstar>1e7))[0]
	hid_all = hid_all[use]

	use_bh = np.where(np.in1d(hid_bh, hid_all))[0]
	hid_bh = hid_bh[use_bh]

	# cross check with resolved halos
	use_bh_any = np.where(np.in1d(hid_bh_any, hid_all))[0]
	hid_bh_any = hid_bh_any[use_bh_any]

	if not match_limit:
		match_limit = 4

	new_mstar_all = []
	new_mstar_bh = []
	new_mstar_bh_any = []

	if use_con:
		ok_to_use = np.where(matched['nmatch_all'] >= match_limit)[0]
		key = 'Mstar_all'
	else:
		ok_to_use = np.where(matched['nmatch_mvms'] >= match_limit)[0]
		key = 'Mstar_mvms'
	for i in range(len(hid_all)):
		if hid_all[i] in matched['hid'][ok_to_use]:
			if len(np.where(matched['hid']==hid_all[i])[0])!=1:
				raise RuntimeError("whoa something is weird here!!!!")
			new_mstar_all.append(matched[key][ok_to_use[(matched['hid'][ok_to_use] == hid_all[i])]])
		else: continue
	for i in range(len(hid_bh)):
		if hid_bh[i] in matched['hid'][ok_to_use]:
			if len(np.where(matched['hid'] == hid_bh[i])[0]) != 1:
				raise RuntimeError("whoa something is weird here!!!!")
			new_mstar_bh.append(matched[key][ok_to_use[(matched['hid'][ok_to_use] == hid_bh[i])]])
		else: continue
	for i in range(len(hid_bh_any)):
		if hid_bh_any[i] in matched['hid'][ok_to_use]:
			if len(np.where(matched['hid'] == hid_bh_any[i])[0]) != 1:
				raise RuntimeError("whoa something is weird here!!!!")
			new_mstar_bh_any.append(matched[key][ok_to_use[(matched['hid'][ok_to_use] == hid_bh_any[i])]])
		else: continue

	new_mstar_all = np.array(new_mstar_all)
	new_mstar_bh = np.array(new_mstar_bh)
	new_mstar_bh_any = np.array(new_mstar_bh_any)

	occ_frac = np.zeros(len(mstar_bins)-1)
	occ_frac_any = np.zeros(len(mstar_bins) - 1)
	occ_frac_err = np.zeros((len(mstar_bins)-1, 2))
	occ_frac_any_err = np.zeros((len(mstar_bins) - 1, 2))

	frac_kept_all = len(new_mstar_all)/len(hid_all)
	frac_kept_bh = len(new_mstar_bh)/len(hid_bh)
	frac_kept_bh_any = len(new_mstar_bh)/len(hid_bh_any)

	print(len(hid_all), len(new_mstar_all), len(hid_bh), len(new_mstar_bh))

	for i in range(len(mstar_bins) - 1):
		obin_all = np.where((np.log10(new_mstar_all * star_corr) > mstar_bins[i]) &
		                       (np.log10(new_mstar_all * star_corr) < mstar_bins[i + 1]))[0]
		obin_bh = np.where((np.log10(new_mstar_bh * star_corr) > mstar_bins[i]) &
		                      (np.log10(new_mstar_bh * star_corr) < mstar_bins[i + 1]))[0]
		obin_bh_any = np.where((np.log10(new_mstar_bh_any * star_corr) > mstar_bins[i]) &
		                      (np.log10(new_mstar_bh_any * star_corr) < mstar_bins[i + 1]))[0]
		if len(obin_all) > 0:
			occ_frac_err[i] = binomial_errors(0.95, len(obin_bh), len(obin_all))
			occ_frac[i] = np.float(len(obin_bh)) / len(obin_all)
			occ_frac_any_err[i] = binomial_errors(0.95, len(obin_bh_any), len(obin_all))
			occ_frac_any[i] = np.float(len(obin_bh_any)) / len(obin_all)
		else:
			occ_frac[i] = np.nan

	notes = "binomial errors, matched based on Mstar, Mvir"
	if use_con:
		notes = notes+", and concentration (Vmax/V200)"

	notes = notes+ " and "+str(frac_kept_all*100)+"% halos are kept, "+str(frac_kept_bh*100)+"% with central BHs"

	data = {'mstar_bins': mstar_bins,
	        'of_all': occ_frac,
	        'of_all_err': occ_frac_err,
	        'of_any': occ_frac_any,
	        'of_any_err': occ_frac_any_err,
	        'notes': notes}

	return data