"""
plot_heat_maps.py
Plotting of evolutionary trajectories on host-outcome measurements.

author: Cameron Smith
based from: https://github.com/CameronSmith50/Defensive-Symbiosis/blob/main/Virulence_Coevo.py
"""

#%%

from modeling import *
import os
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from scipy.ndimage import gaussian_filter
import pickle as pkl
import sys

#%%

def plot2HeatMaps(
        dirPath,
        separate = True,
        together = True
):
    """
    Plots heat maps based on data in a directory. The directory is specified in
    dirPath. If separate is true, will create and save a separate plot for each
    (c1,c2) pair. If together is on, will generate a figure with all values together.
    """

    # Check if the directory exists
    if os.path.isdir(dirPath):
        listOfDirs = [dirPath + ii + '/' for ii in os.listdir(dirPath)]
        nDir = len(listOfDirs)
    else:
        raise Exception('File does not exist')
    
    saveFigLoc = '/'.join([ii if ii != 'data' else 'results' for ii in dirPath.split('/')])
    if os.path.isdir(saveFigLoc):
        rePlot = input('Plots already generated. Would you like to regenerate? [yes (y)/no (n)]: ')
    
    if os.path.isdir(saveFigLoc) != True or rePlot == 'y':
    
        if os.path.isdir(saveFigLoc) != True:
            os.mkdir(saveFigLoc)

        # Calculate the number of (c1, c2) pairs. This will assume that we have
        # 3 y0 values per pair.
        nPlot = int(nDir/3)

        # Find the values of c1 and c2
        c1vals = []
        c2vals = []
        y0vals = []
        for ii, direct in enumerate(listOfDirs):
            with open(direct + 'parameters.txt') as f:
                lines = [kk for kk in f]
                c1vals.append(float(lines[3][3:-1]))
                c2vals.append(float(lines[6][3:-1]))
                y0vals.append(float(lines[2][11:-1]))
                if ii == 0:
                    ode_parameters = {}
                    for jj in range(1, len(lines)):
                        if jj < len(lines)-1:
                            key, val = lines[jj][:-1].split('=')
                        else:
                            key, val = lines[jj].split('=')
                        ode_parameters[key] = float(val)
        
        xTemp = [[c1vals[ii], c2vals[ii]] for ii in range(len(c1vals))]
        yTemp = []
        for elem in xTemp:
            if elem not in yTemp:
                yTemp.append(elem)
        c1vec = list(np.array(yTemp).transpose()[0])
        c2vec = list(np.array(yTemp).transpose()[1])
        c1Dict = {}
        for jj in range(len(c1vec)):
            c1Dict[str(c1vec[jj])] = jj
        c2Dict = {}
        for jj in range(len(c2vec)):
            c2Dict[str(c2vec[jj])] = jj
        resVec = np.linspace(ode_parameters['resMin'], ode_parameters['resMax'], int(ode_parameters['nRes']))
        virVec = np.linspace(ode_parameters['virMin'], ode_parameters['virMax'], int(ode_parameters['nVir']))

        # Create all of the figures that we need
        if separate:
            sepWidth = 20
            sepHeight = 15
            sepFigs = [plt.figure(figsize=(sepWidth, sepHeight)) for ii in range(nPlot)]
            sepAxs = [[fig.add_subplot(121), fig.add_subplot(122)] for fig in sepFigs]

        if together:
            togWidth = 20*len(c1vec)
            togHeight = 15*len(c2vec)
            togFig = plt.figure(figsize=(togWidth, togHeight))
            togAxs = []
            for ii in range(2*len(c1vec)*len(c2vec)):
                togAxs.append(togFig.add_subplot(len(c1vec), 2*len(c2vec), ii+1))

        # Colourmaps
        cmap = ListedColormap(['#ef8a62','#fddbc7','#d1e5f0','#67a9cf'])
        cmap.set_under('#b2182b')
        cmap.set_over('#2166ac')
        cmapalt = ListedColormap(['#af8dc3','#e7d4e8','#d9f0d3','#7fbf7b'])
        cmapalt.set_under('#762a83')
        cmapalt.set_over('#1b7837')
        mpl.rcParams['contour.negative_linestyle'] = 'solid'

        # Add the heatmaps to each plot
        levels = [ii*10 for ii in range(-7,8)]
        for c1Ind, c1 in enumerate(c1vec):
            ode_parameters['c1'] = c1
            ode_parameters['c2'] = c2vec[c1Ind]
            NMat = np.zeros((int(ode_parameters['nVir']), int(ode_parameters['nRes'])))
            deathRateMat = np.zeros((int(ode_parameters['nVir']), int(ode_parameters['nRes'])))
            for resInd, res in enumerate(resVec):
                for virInd, vir in enumerate(virVec):
                    ode_parameters['alpha_P'] = vir
                    ode_parameters['resistance'] = res
                    AHstar, APstar = compute_ancestral_steady_state(ode_parameters)
                    Avir = compute_ancestral_virulence(ode_parameters)
                    Ht, Dt, Pt, Bt = run_ode_solver(ode_parameters, make_system(ode_parameters))
                    StSt = list(approximate_steady_state(Ht, Dt, Pt, Bt).values())
                    NMat[virInd, resInd] = np.sum(StSt)/(AHstar+APstar) - 1
                    deathRateMat[virInd, resInd] = 1 - ((ode_parameters['alpha_D']*StSt[1] + vir*StSt[2] + build_alpha_B()(ode_parameters['alpha_D'], vir)*StSt[3])/np.sum(StSt))/(Avir*APstar/(AHstar+APstar))
            if separate:
                sepAxs[c1Ind][0].contourf(resVec, virVec, gaussian_filter(100*NMat, 1), extend='both', cmap=cmap, alpha=0.3, levels=levels)
                cont = sepAxs[c1Ind][0].contour(resVec, virVec, gaussian_filter(100*NMat, 1), colors='dimgray', levels=levels)
                sepAxs[c1Ind][0].clabel(cont, inline=True, fontsize=18)
                sepAxs[c1Ind][0].plot([resVec[0], resVec[-1]], [Avir, Avir], 'k--')
                sepAxs[c1Ind][1].contourf(resVec, virVec, gaussian_filter(100*deathRateMat, 1), extend='both', cmap=cmap, alpha=0.3, levels=levels)
                cont = sepAxs[c1Ind][1].contour(resVec, virVec, gaussian_filter(100*deathRateMat, 1), colors='dimgray', levels=levels)
                sepAxs[c1Ind][1].clabel(cont, inline=True, fontsize=18)
                sepAxs[c1Ind][1].plot([resVec[0], resVec[-1]], [Avir, Avir], 'k--')
            if together:
                plotInd = 2*c1Ind
                togAxs[plotInd].contourf(resVec, virVec, gaussian_filter(100*NMat, 1), extend='both', cmap=cmap, alpha=0.3, levels=levels)
                cont = togAxs[plotInd].contour(resVec, virVec, gaussian_filter(100*NMat, 1), colors='dimgray', levels=levels)
                togAxs[plotInd].clabel(cont, inline=True, fontsize=18)
                togAxs[plotInd].plot([resVec[0], resVec[-1]], [Avir, Avir], 'k--')
                togAxs[plotInd+1].contourf(resVec, virVec, gaussian_filter(100*deathRateMat, 1), extend='both', cmap=cmap, alpha=0.3, levels=levels)
                cont = togAxs[plotInd+1].contour(resVec, virVec, gaussian_filter(100*deathRateMat, 1), colors='dimgray', levels=levels)
                togAxs[plotInd+1].clabel(cont, inline=True, fontsize=18)
                togAxs[plotInd+1].plot([resVec[0], resVec[-1]], [Avir, Avir], 'k--')

        # Now add trajectories
        for ii, direct in enumerate(listOfDirs):
            file = open(direct + 'branches.pkl', 'rb')
            pklDict = pkl.load(file)
            file.close()
            c1Ind = c1Dict[str(c1vals[ii])]
            c2Ind = c2Dict[str(c2vals[ii])]
            resLower = pklDict['resLower']
            resUpper = pklDict['resUpper']
            virLower = pklDict['virLower']
            virUpper = pklDict['virUpper']
            Dext = np.any(np.isnan(resLower))
            Pext = np.any(np.isnan(virLower))
            if separate:
                if Dext:
                    sepAxs[c1Ind][0].plot(y0vals[ii], Avir, 'r+', ms=20)
                    sepAxs[c1Ind][1].plot(y0vals[ii], Avir, 'r+', ms=20)
                elif Pext:
                    sepAxs[c1Ind][0].plot(y0vals[ii], Avir, 'rx', ms=20)
                    sepAxs[c1Ind][1].plot(y0vals[ii], Avir, 'rx', ms=20)
                else:
                    sepAxs[c1Ind][0].plot(resLower, virLower, 'k', lw=2)
                    # sepAxs[c1Ind][0].plot(resUpper, virUpper, 'k', lw=2)
                    sepAxs[c1Ind][0].plot(resLower[0], virLower[0], 'r.', ms=10)
                    sepAxs[c1Ind][0].plot(resLower[-1], virLower[-1], 'g.', ms=10)
                    sepAxs[c1Ind][1].plot(resLower, virLower, 'k', lw=2)
                    # sepAxs[c1Ind][1].plot(resUpper, virUpper, 'k', lw=2)
                    sepAxs[c1Ind][1].plot(resUpper[0], virUpper[0], 'r.', ms=10)
                    # sepAxs[c1Ind][1].plot(resUpper[-1], virUpper[-1], 'g.', ms=10)
            if together:
                plotInd = 2*c1Ind
                if Dext:
                    togAxs[plotInd].plot(y0vals[ii], Avir, 'r+', ms=20)
                    togAxs[plotInd+1].plot(y0vals[ii], Avir, 'r+', ms=20)
                elif Pext:
                    togAxs[plotInd].plot(y0vals[ii], Avir, 'rx', ms=20)
                    togAxs[plotInd+1].plot(y0vals[ii], Avir, 'rx', ms=20)
                else:
                    togAxs[plotInd].plot(resLower, virLower, 'k', lw=2)
                    # togAxs[plotInd].plot(resUpper, virUpper, 'k', lw=2)
                    togAxs[plotInd].plot(resLower[0], virLower[0], 'r.', ms=10)
                    # togAxs[plotInd].plot(resUpper[-1], virUpper[-1], 'g.', ms=10)
                    togAxs[plotInd+1].plot(resLower, virLower, 'k', lw=2)
                    # togAxs[plotInd+1].plot(resUpper, virUpper, 'k', lw=2)
                    togAxs[plotInd+1].plot(resLower[0], virLower[0], 'r.', ms=10)
                    # togAxs[plotInd+1].plot(resUpper[-1], virUpper[-1], 'g.', ms=10)

        # Save all the figures
        if separate:
            for ii in range(len(sepFigs)):
                sepFigs[ii].savefig(saveFigLoc + 'sepFig_' + str(ii) + '.png', bbox_inches = 'tight')
                sepFigs[ii].savefig(saveFigLoc + 'sepFig_' + str(ii) + '.pdf', bbox_inches = 'tight')

        if together:
            togFig.savefig(saveFigLoc + 'togFig.png', bbox_inches = 'tight')
            togFig.savefig(saveFigLoc + 'togFig.pdf', bbox_inches = 'tight')

def plotHeatMaps(
        dirPath,
        separate = True,
        together = True
):
    """
    Plots heat maps based on data in a directory. The directory is specified in
    dirPath. If separate is true, will create and save a separate plot for each
    (c1,c2) pair. If together is on, will generate a figure with all values together.
    """

    # Check if the directory exists
    if os.path.isdir(dirPath):
        listOfDirs = [dirPath + ii + '/' for ii in os.listdir(dirPath)]
        nDir = len(listOfDirs)
    else:
        raise Exception('File does not exist')
    
    saveFigLoc = '/'.join([ii if ii != 'data' else 'results' for ii in dirPath.split('/')])
    if os.path.isdir(saveFigLoc):
        rePlot = input('Plots already generated. Would you like to regenerate? [yes (y)/no (n)]: ')
    
    if os.path.isdir(saveFigLoc) != True or rePlot == 'y':
    
        if os.path.isdir(saveFigLoc) != True:
            os.mkdir(saveFigLoc)

        # Calculate the number of (c1, c2) pairs. This will assume that we have
        # 3 y0 values per pair.
        nPlot = int(nDir/3)
        nPlotDir = int(np.ceil(np.sqrt(nPlot)))

        # Find the values of c1 and c2
        c1vals = []
        c2vals = []
        y0vals = []
        for ii, direct in enumerate(listOfDirs):
            with open(direct + 'parameters.txt', 'r') as f:
                lines = [kk for kk in f]
                c1vals.append(float(lines[3][3:-1]))
                c2vals.append(float(lines[6][3:-1]))
                y0vals.append(float(lines[2][11:-1]))
                if ii == 0:
                    ode_parameters = {}
                    for jj in range(1, len(lines)):
                        if jj < len(lines)-1:
                            key, val = lines[jj][:-1].split('=')
                        else:
                            key, val = lines[jj].split('=')
                        ode_parameters[key] = float(val)
        
        xTemp = [[c1vals[ii], c2vals[ii]] for ii in range(len(c1vals))]
        yTemp = []
        for elem in xTemp:
            if elem not in yTemp:
                yTemp.append(elem)
        c1vec = list(np.array(yTemp).transpose()[0])
        c2vec = list(np.array(yTemp).transpose()[1])
        c1Dict = {}
        for jj in range(len(c1vec)):
            c1Dict[str(c1vec[jj])] = jj
        c2Dict = {}
        for jj in range(len(c2vec)):
            c2Dict[str(c2vec[jj])] = jj
        resVec = np.linspace(ode_parameters['resMin'], ode_parameters['resMax'], int(ode_parameters['nRes']))
        virVec = np.linspace(ode_parameters['virMin'], ode_parameters['virMax'], int(ode_parameters['nVir']))

        # Create all of the figures that we need
        if separate:
            sepWidth = 20
            sepHeight = 15
            sepFigs = [plt.figure(figsize=(sepWidth, sepHeight)) for ii in range(nPlot)]
            sepAxs = [fig.add_subplot(111) for fig in sepFigs]

        if together:
            togWidth = 20*nPlotDir
            togHeight = 15*nPlotDir
            togFig = plt.figure(figsize=(togWidth, togHeight))
            togAxs = []
            for ii in range(len(c1vec)):
                togAxs.append(togFig.add_subplot(nPlotDir, nPlotDir, ii+1))

        fsize = 45
        mpl.rcParams['font.size'] = fsize

        # Colourmaps
        cmap = ListedColormap(['#ef8a62','#fddbc7','#d1e5f0','#67a9cf'])
        cmap.set_under('#b2182b')
        cmap.set_over('#2166ac')
        cmapalt = ListedColormap(['#af8dc3','#e7d4e8','#d9f0d3','#7fbf7b'])
        cmapalt.set_under('#762a83')
        cmapalt.set_over('#1b7837')
        mpl.rcParams['contour.negative_linestyle'] = 'solid'

        # Add the heatmaps to each plot
        levels = [ii*10 for ii in range(-7,8)]
        Avir = compute_ancestral_virulence(ode_parameters)
        for c1Ind, c1 in enumerate(c1vec):
            ode_parameters['c1'] = c1
            ode_parameters['c2'] = c2vec[c1Ind]
            NMat = np.zeros((int(ode_parameters['nVir']), int(ode_parameters['nRes'])))
            for resInd, res in enumerate(resVec):
                for virInd, vir in enumerate(virVec):
                    ode_parameters['alpha_P'] = vir
                    ode_parameters['resistance'] = res
                    AHstar, APstar = compute_ancestral_steady_state(ode_parameters)
                    Ht, Dt, Pt, Bt = run_ode_solver(ode_parameters, make_system(ode_parameters))
                    StSt = list(approximate_steady_state(Ht, Dt, Pt, Bt).values())
                    NMat[virInd, resInd] = np.sum(StSt)/(AHstar+APstar) - 1
            if separate:
                sepAxs[c1Ind].contourf(resVec, virVec, gaussian_filter(100*NMat, 1), extend='both', cmap=cmap, alpha=0.3, levels=levels)
                cont = sepAxs[c1Ind].contour(resVec, virVec, gaussian_filter(100*NMat, 1), colors='dimgray', levels=levels)
                sepAxs[c1Ind].clabel(cont, inline=True, fontsize=fsize)
                sepAxs[c1Ind].plot([resVec[0], resVec[-1]], [Avir, Avir], 'k--')
                sepAxs[c1Ind].set_title(f'c1 = {c1}, c2 = {c2vec[c1Ind]}')
            if together:
                plotInd = c1Ind
                togAxs[plotInd].contourf(resVec, virVec, gaussian_filter(100*NMat, 1), extend='both', cmap=cmap, alpha=0.3, levels=levels)
                cont = togAxs[plotInd].contour(resVec, virVec, gaussian_filter(100*NMat, 1), colors='dimgray', levels=levels)
                togAxs[plotInd].clabel(cont, inline=True, fontsize=fsize)
                togAxs[plotInd].plot([resVec[0], resVec[-1]], [Avir, Avir], 'k--')
                    
        # Now add trajectories
        for ii, direct in enumerate(listOfDirs):
            file = open(direct + 'branches.pkl', 'rb')
            pklDict = pkl.load(file)
            file.close()
            jj = int(np.floor(ii/3))
            c1Ind = jj
            c2Ind = jj
            resLower = pklDict['resLower']
            resUpper = pklDict['resUpper']
            virLower = pklDict['virLower']
            virUpper = pklDict['virUpper']
            Dext = np.any(np.isnan(resLower))
            Pext = np.any(np.isnan(virLower))
            if separate:
                if Dext:
                    sepAxs[c1Ind].plot(y0vals[ii], Avir, 'g+', ms=40)
                elif Pext:
                    sepAxs[c1Ind].plot(y0vals[ii], Avir, 'gx', ms=40)
                else:
                    sepAxs[c1Ind].plot(resLower, virLower, 'k', lw=4)
                    # sepAxs[c1Ind].plot(resUpper, virUpper, 'k', lw=4)
                    sepAxs[c1Ind].plot(resLower[0], virLower[0], 'g.', ms=40)
                    sepAxs[c1Ind].plot(resLower[-1], virLower[-1], 'r.', ms=40)
                    # sepAxs[c1Ind].plot(resUpper[-1], virUpper[-1], 'r.', ms=40)
            if together:
                plotInd = c1Ind
                if Dext:
                    togAxs[plotInd].plot(y0vals[ii], Avir, 'g+', ms=40)
                elif Pext:
                    togAxs[plotInd].plot(y0vals[ii], Avir, 'gx', ms=40)
                else:
                    togAxs[plotInd].plot(resLower, virLower, 'k', lw=4)
                    # togAxs[plotInd].plot(resUpper, virUpper, 'k', lw=4)
                    togAxs[plotInd].plot(resLower[0], virLower[0], 'g.', ms=40)
                    togAxs[plotInd].plot(resLower[-1], virLower[-1], 'r.', ms=40)
                    # togAxs[plotInd].plot(resUpper[-1], virUpper[-1], 'r.', ms=40)

        # Alphabet
        alph = 'ABCDEFGHIJKLM'

        # Fix all of the axis labels
        togAxs[0].set_title(rf'$c_1 = ${c1vec[0]:.2f}, $c_2 = ${c2vec[0]:.2f}', fontsize=fsize)
        togAxs[0].set_xlabel('')
        togAxs[0].set_xticks([0, 0.5, 1.0])
        togAxs[0].set_xticklabels([])
        togAxs[0].set_ylabel(r'Parasite virulence, $\alpha_P$', fontsize=fsize)
        togAxs[0].set_yticks([0, 0.5, 1.0])
        togAxs[0].set_yticklabels(['0', '0.5', '1'], fontsize=fsize)

        togAxs[1].set_title(rf'$c_1 = ${c1vec[1]:.2f}, $c_2 = ${c2vec[1]:.2f}', fontsize=fsize)
        togAxs[1].set_xlabel('')
        togAxs[1].set_xticks([0, 0.5, 1.0])
        togAxs[1].set_xticklabels([])
        togAxs[1].set_ylabel('')
        togAxs[1].set_yticks([0, 0.5, 1.0])
        togAxs[1].set_yticklabels([])

        togAxs[2].set_title(rf'$c_1 = ${c1vec[2]:.2f}, $c_2 = ${c2vec[2]:.2f}', fontsize=fsize)
        togAxs[2].set_xlabel(r'Resistance, $y$', fontsize=fsize)
        togAxs[2].set_xticks([0, 0.5, 1.0])
        togAxs[2].set_xticklabels(['0', '0.5', '1'], fontsize=fsize)
        togAxs[2].set_ylabel(r'Parasite virulence, $\alpha_P$', fontsize=fsize)
        togAxs[2].set_yticks([0, 0.5, 1.0])
        togAxs[2].set_yticklabels(['0', '0.5', '1'], fontsize=fsize)

        togAxs[3].set_title(rf'$c_1 = ${c1vec[3]:.2f}, $c_2 = ${c2vec[3]:.2f}', fontsize=fsize)
        togAxs[3].set_xlabel(r'Resistance, $y$', fontsize=fsize)
        togAxs[3].set_xticks([0, 0.5, 1.0])
        togAxs[3].set_xticklabels(['0', '0.5', '1'], fontsize=fsize)
        togAxs[3].set_ylabel('')
        togAxs[3].set_yticks([0, 0.5, 1.0])
        togAxs[3].set_yticklabels([])

        # Add letters
        togAxs[0].text(0.0, 1.03, 'A.')
        togAxs[1].text(0.0, 1.03, 'B.')
        togAxs[2].text(0.0, 1.03, 'C.')
        togAxs[3].text(0.0, 1.03, 'D.')

        # Save all the figures
        if separate:
            for ii in range(len(sepFigs)):
                sepFigs[ii].savefig(saveFigLoc + 'sepFig_' + str(ii) + '.png', bbox_inches = 'tight')
                sepFigs[ii].savefig(saveFigLoc + 'sepFig_' + str(ii) + '.pdf', bbox_inches = 'tight')

        if together:
            togFig.savefig(saveFigLoc + 'togFig.png', bbox_inches = 'tight')
            togFig.savefig(saveFigLoc + 'togFig.pdf', bbox_inches = 'tight')

def plotHeatMapsandTraj(dirPath):

    # Check if the directory exists
    if os.path.isdir(dirPath):
        listOfDirs = [dirPath + ii + '/' for ii in os.listdir(dirPath)]
        nDir = len(listOfDirs)
    else:
        raise Exception('File does not exist')
    
    saveFigLoc = '/'.join([ii if ii != 'data' else 'results' for ii in dirPath.split('/')])
    if os.path.isdir(saveFigLoc):
        rePlot = input('Plots already generated. Would you like to regenerate? [yes (y)/no (n)]: ')
    
    if os.path.isdir(saveFigLoc) != True or rePlot == 'y':
    
        if os.path.isdir(saveFigLoc) != True:
            os.mkdir(saveFigLoc)

        # Calculate the number of (c1, c2) pairs. This will assume that we have
        # 3 y0 values per pair.
        nPlot = int(nDir/3)

        # Find the values of c1 and c2
        c1vals = []
        c2vals = []
        y0vals = []
        for ii, direct in enumerate(listOfDirs):
            with open(direct + 'parameters.txt', 'r') as f:
                lines = [kk for kk in f]
                c1vals.append(float(lines[3][3:-1]))
                c2vals.append(float(lines[6][3:-1]))
                y0vals.append(float(lines[2][11:-1]))
                if ii == 0:
                    ode_parameters = {}
                    for jj in range(1, len(lines)):
                        if jj < len(lines)-1:
                            key, val = lines[jj][:-1].split('=')
                        else:
                            key, val = lines[jj].split('=')
                        ode_parameters[key] = float(val)
        
        xTemp = [[c1vals[ii], c2vals[ii]] for ii in range(len(c1vals))]
        yTemp = []
        for elem in xTemp:
            if elem not in yTemp:
                yTemp.append(elem)
        c1vec = list(np.array(yTemp).transpose()[0])
        c2vec = list(np.array(yTemp).transpose()[1])
        c1Dict = {}
        for jj in range(len(c1vec)):
            c1Dict[str(c1vec[jj])] = jj
        c2Dict = {}
        for jj in range(len(c2vec)):
            c2Dict[str(c2vec[jj])] = jj
        resVec = np.linspace(ode_parameters['resMin'], ode_parameters['resMax'], int(ode_parameters['nRes']))
        virVec = np.linspace(ode_parameters['virMin'], ode_parameters['virMax'], int(ode_parameters['nVir']))

        # Set up the figure
        figWidth = 50
        figHeight = 30
        fig, axs = plt.subplots(ncols = nPlot, nrows = 3, figsize=(figWidth, figHeight))
        fsize = 45
        mpl.rcParams['font.size'] = fsize

        # Set up the axes
        axD = []
        axH = []
        for ii in range(nPlot):
            gsD = axs[0, ii].get_gridspec()
            gsH = axs[2, ii].get_gridspec()
            axD.append(fig.add_subplot(gsD[:2, ii]))
            axH.append(fig.add_subplot(gsH[2, (ii):((ii+1))]))

        # Remove all the previous axes
        for axList in axs:
            for ax in axList:
                ax.remove()

        # Colourmaps
        cmap = ListedColormap(['#ef8a62','#fddbc7','#d1e5f0','#67a9cf'])
        cmap.set_under('#b2182b')
        cmap.set_over('#2166ac')
        cmapalt = ListedColormap(['#af8dc3','#e7d4e8','#d9f0d3','#7fbf7b'])
        cmapalt.set_under('#762a83')
        cmapalt.set_over('#1b7837')
        mpl.rcParams['contour.negative_linestyle'] = 'solid'

        # Add the heatmaps
        levels = [ii*10 for ii in range(-7,8)]
        Avir = compute_ancestral_virulence(ode_parameters)
        virVecHM = np.linspace(0.5, 0.7, 51)
        for c1Ind, c1 in enumerate(c1vec):
            ode_parameters['c1'] = c1
            ode_parameters['c2'] = c2vec[c1Ind]
            NMat = np.zeros((int(ode_parameters['nVir']), int(ode_parameters['nRes'])))
            for resInd, res in enumerate(resVec):
                for virInd, vir in enumerate(virVecHM):
                    ode_parameters['alpha_P'] = vir
                    ode_parameters['resistance'] = res
                    AHstar, APstar = compute_ancestral_steady_state(ode_parameters)
                    Ht, Dt, Pt, Bt = run_ode_solver(ode_parameters, make_system(ode_parameters))
                    StSt = list(approximate_steady_state(Ht, Dt, Pt, Bt).values())
                    NMat[virInd, resInd] = np.sum(StSt)/(AHstar+APstar) - 1
            axH[c1Ind].contourf(resVec, virVecHM, gaussian_filter(100*NMat, 1), extend='both', cmap=cmap, alpha=0.3, levels=levels)
            cont = axH[c1Ind].contour(resVec, virVecHM, gaussian_filter(100*NMat, 1), colors='dimgray', levels=levels)
            axH[c1Ind].clabel(cont, inline=True, fontsize=18)
            axH[c1Ind].plot([resVec[0], resVec[-1]], [Avir, Avir], 'k--')

            # Now add trajectories
            for ii, direct in enumerate(listOfDirs):
                file = open(direct + 'branches.pkl', 'rb')
                pklDict = pkl.load(file)
                file.close()
                jj = int(np.floor(ii/3))
                resLower = pklDict['resLower']
                resUpper = pklDict['resUpper']
                virLower = pklDict['virLower']
                virUpper = pklDict['virUpper']
                Dext = np.any(np.isnan(resLower))
                Pext = np.any(np.isnan(virLower))
                if Dext:
                    axH[jj].plot(y0vals[ii], Avir, 'g+', ms=40)
                elif Pext:
                    axH[jj].plot(y0vals[ii], Avir, 'gx', ms=40)
                else:
                    axH[jj].plot(resLower, virLower, 'k', lw=4)
                    # axH[jj].plot(resUpper, virUpper, 'k', lw=4)
                    axH[jj].plot(resLower[0], virLower[0], 'r.', ms=40)
                    axH[jj].plot(resLower[-1], virLower[-1], 'r.', ms=40)
                    # axH[jj].plot(resUpper[-1], virUpper[-1], 'r.', ms=40)

            # Create the trajectory plots
            DDCoEx = []
            DDPExt = []
            DDDExt = []

            # Loop through the list of directories and check for if both c1 and c2 match
            for kk, direct in enumerate(listOfDirs):
                if np.abs(c1vals[kk] - c1) < 1e-5 and np.abs(c2vals[kk] - c2vec[c1Ind]) < 1e-5:
                    Dvals = np.loadtxt(direct + 'def_symbiont_freqs_from_coevo_sim.csv', delimiter=',')
                    Pvals = np.loadtxt(direct + 'parasite_freqs_from_coevo_sim.csv', delimiter=',')
                    maskD = np.tile(Dvals.sum(axis=1) < 1e-5, (len(resVec), 1)).transpose()
                    maskP = np.tile(Pvals.sum(axis=1) < 1e-5, (len(virVec), 1)).transpose()
                    Dvals[maskP] = np.nan
                    DDCoEx.append(Dvals)
                    Dvals = np.loadtxt(direct + 'def_symbiont_freqs_from_coevo_sim.csv', delimiter=',')
                    Dvals[~maskP] = np.nan
                    DDPExt.append(Dvals)
                    Dvals = np.loadtxt(direct + 'def_symbiont_freqs_from_coevo_sim.csv', delimiter=',')
                    Dvals[~maskD] = np.nan
                    DDDExt.append(Dvals) 
            
            # Sum these matrices
            DMatCoEx = np.zeros(DDCoEx[0].shape)
            DMatPExt = np.zeros(DDPExt[0].shape)
            DMatDExt = np.zeros(DDDExt[0].shape)
            for ii in range(3):
                DMatCoEx = np.nansum(np.stack((DMatCoEx,DDCoEx[ii])), axis=0)
                DMatPExt = np.nansum(np.stack((DMatPExt,DDPExt[ii])), axis=0)
                DMatDExt = np.nansum(np.stack((DMatDExt,DDDExt[ii])), axis=0)
            DMatPExt[DMatPExt<1e-5] = np.nan
            DMatDExt[DMatDExt<1e-5] = np.nan

            # Add the plots
            # Defensive symbiont
            axD[c1Ind].pcolormesh(resVec, np.arange(len(DMatCoEx)), DMatCoEx, cmap='Greys')
            axD[c1Ind].pcolormesh(resVec, np.arange(len(DMatPExt)), DMatPExt, cmap='Reds')
            axD[c1Ind].pcolormesh(resVec, np.arange(len(DMatDExt)), DMatDExt, cmap='Greys')

        # Alphabet
        alph = 'ABCDEFGHIJKLM'
        
        # Fix the axes and add labels
        for ii in range(nPlot):
            axD[ii].set_title(rf'$c_1$ = {c1vec[ii]}, $c_2$ = {c2vec[ii]}', fontsize=fsize)
            axD[ii].set_xticks([0, 0.5, 1])
            axD[ii].set_xticklabels(['0', '0.5', '1'])
            axD[ii].set_yticks([0, (len(DMatCoEx)-1)/2, len(DMatCoEx)-1])
            axD[ii].set_yticklabels(['0', str(int((len(DMatCoEx)-1)/2)), str(int(len(DMatCoEx)-1))], fontsize=fsize)
            axD[ii].set_xticklabels([])
            axH[ii].set_xticks([0, 0.5, 1])
            axH[ii].set_xticklabels(['0', '0.5', '1'])
            axH[ii].set_yticks([0.5, 0.6, 0.7])
            axH[ii].set_yticklabels(['0.5', '0.6', '0.7'], fontsize=fsize)
            axD[ii].text(-0.05, 1.05*len(DMatCoEx), f'{alph[ii]}.')
            if ii != 0:
                axD[ii].set_yticklabels([])
                axH[ii].set_yticklabels([])
            else:
                pass

        # Add the axis labels
        fig.text(0.51, 0.075, r'Resistance, $y$', ha='center', fontsize=fsize)
        axD[0].set_ylabel('Evolutionary time', fontsize=fsize)
        axH[0].set_ylabel(r'Parasite virulence, $\alpha_P$', fontsize=fsize)
            
        fig.savefig(saveFigLoc + 'withTraj.png', bbox_inches='tight')
        # plt.show()

if __name__ == '__main__':

    if len(sys.argv) == 1:
        print('Please specify a directory containing data')

    elif len(sys.argv) == 2:
        plotHeatMaps(f'./data/{sys.argv[1]}')

    else:
        print('Too many input arguments')