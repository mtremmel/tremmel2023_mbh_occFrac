import numpy as np
import Simpy

def identify_sats_geha(cen,rvir,mstar,mvir,hid,scale=1):
        sat_flag = np.zeros(len(rvir))

        for i in range(len(rvir)):
                relpos = cen[i] - cen
                Simpy.util.wrap(relpos,scale)
                d = np.sqrt(np.sum(relpos**2,axis=1))
                if mstar[i]>1e10:
                        if len(np.where((d<rvir)&(mvir>mvir[i])&(hid!=hid[i]))[0]) > 0:
                                sat_flag[i] = 1
                else:
                        if len(np.where(((d<1500)&(mstar>2.5e10))|((d<rvir)&(mstar>0.25*mstar[i])&(hid!=hid[i])))[0])>0:
                                sat_flag[i] = 1
        return sat_flag

def find_closest_host(cen,rvir,mstar,mvir,hid,scale=1, min_host_dist=2000):
    mclose = np.ones(len(rvir))*-1
    Dclose = np.ones(len(rvir))*-1
    hid_close = np.ones(len(rvir))*-1

    mhost = np.ones(len(rvir)) * -1
    Dhost = np.ones(len(rvir)) * -1
    hid_host = np.ones(len(rvir)) * -1

    for i in range(len(rvir)):

        relpos = cen[i] - cen
        Simpy.util.wrap(relpos, scale)
        d = np.sqrt(np.sum(relpos ** 2, axis=1))

        #find halos where target is within 2xRvir
        ihost = np.where((mvir>mvir[i])&(hid!=hid[i])&(d<2*rvir))[0]
        if len(ihost)>0:
            ihost_max = ihost[np.argmax(mvir[ihost])] #max host mass
            mhost[i] = mvir[ihost_max]
            hid_host[i] = hid[ihost_max]
            Dhost[i] = d[ihost_max]

        #find the mass of the closest massive halo
        iclose = np.where((mvir>mvir[i]) & (d < min_host_dist) & (hid!=hid[i]))[0]
        if len(iclose)>0:
            imassive = iclose[np.argmax(mvir[iclose])]
            mclose[i] = mvir[imassive]
            Dclose[i] = d[imassive]
            hid_close[i] = hid[imassive]

    return mhost, Dhost, hid_host, mclose, Dclose, hid_close

def L_edd(mass):
    #input: mass in units of Msol
    #assuming ionized hydrogen (https://en.wikipedia.org/wiki/Eddington_luminosity)
    import pynbody
    lbol_sun = 3.9e33 #solar luminosity ergs/s
    return pynbody.array.SimArray(mass*lbol_sun * 3.2e4, 'erg s**-1')

def calc_bol_lum(mdot, bhmass, erad=0.1):
    #using the equations from Churazov 2005 (also see Habouzit 2022)
    #Eddington ratios calculated assuming constant radiative efficiency
    #Not fully self consistent
    import pynbody
    csq = pynbody.array.SimArray((2.998e10) ** 2, 'erg g**-1')
    lum_sim = mdot.in_units('g s**-1') * csq * erad #raw energy output assumed in simulation (constant eff)
    fedd = lum_sim/L_edd(bhmass)

    lum_final = np.copy(lum_sim)
    try:
        lum_final[(fedd<0.1)] *= 10*fedd[(fedd<0.1)]
    except:
        if hasattr(fedd,'__len__'):
            if len(fedd)>1:
                raise RuntimeError("fedd has weird shape!")
            fedd = fedd[0]
        if fedd<0.1:
            lum_final *= 10*fedd
    return lum_final

def calc_bol_corr_xray(lbol, band='soft'):
    #based on shen 2020 (https://arxiv.org/pdf/2001.02696.pdf)
    lbol_sun = 3.9e33  # solar luminosity ergs/s
    params = {
        "soft": [5.712, -0.026, 17.67, 0.278],
        "hard": [4.073, -0.026, 12.60, 0.278],
    }
    c1, k1, c2, k2 = params[band]
    corr = c1*(lbol/lbol_sun/1e10)**k1 + c2*(lbol/lbol_sun/1e10)**k2
    return 1/corr

def calc_xray_lum(mdot, mass, erad=0.1):
    lbol = calc_bol_lum(mdot, mass, erad)
    bol_corr_total = calc_bol_corr_xray(lbol,'soft') + calc_bol_corr_xray(lbol,'hard')
    return bol_corr_total*lbol

def identify_quench(ms,sfr,z, deltaSF, flat_ssfr=False):
    #quenched fraction at z = 0 based on Romulus main sequence fit
        def MS_Fit_R25_z0(ms):
            return 1.2061331693903192*np.log10(ms) - 11.701074837540167

        quench_flag = np.zeros(len(ms))
        if not flat_ssfr:
                q = np.where(sfr < 10 ** (MS_Fit_R25_z0(ms)-deltaSF))[0]
        else:
                q = np.where(sfr/ms < flat_ssfr)
        quench_flag[q] = 1
        return quench_flag

def binomial_errors(confidence, ntrue, ntot):
    import scipy.stats
    alpha = 1 - confidence
    lo = scipy.stats.beta.ppf(alpha / 2, ntrue, ntot - ntrue + 1)
    hi = scipy.stats.beta.ppf(1 - alpha / 2, ntrue + 1, ntot - ntrue)
    return lo,hi


def get_macc(bhiords, tform1, tform2, step='7779', sim='cosmo25'):
    import tangos as db
    macc = np.ones(len(bhiords)) * -1
    for ii in range(len(bhiords)):
        if ii % 10 == 0: print(ii / len(bhiords))
        iord = bhiords[ii]
        bhdb = db.get_halo(sim+'/%' + str(step) + '/BH_' + str(iord))
        mdot_all = bhdb.calculate('reassemble(BH_mdot_histogram,"sum")')
        tbad = np.concatenate([tform1[ii], tform2[ii]])
        tbad = np.unique(tbad)
        obad = []
        use = np.ones(len(mdot_all))
        tmdot = np.arange(len(mdot_all)) * 0.01
        for j in range(len(tbad)):
            badstep = np.where(np.abs(tmdot - tbad[j]) < 0.01)[0]
            if len(badstep) > 0: obad.extend(badstep)
        iform = np.where(mdot_all > 0)[0][
            0]  # take away first nonzero accretion output (affected by current BH formation)
        if iform not in obad:
            obad.append(iform)
        use[obad] = 0
        use_mdot = np.where(use == 1)[0]
        macc[ii] = np.nansum(mdot_all[use_mdot]) * 0.01 * 1e9

    return macc


