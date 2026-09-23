"""Check the manuscript geometry/wavepackets and directly replot Figure 2(d)."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
import sys,json
import numpy as np
from PIL import Image
root=Path(__file__).resolve().parents[2]
out=root/'code/figure-reproduction/generated/geometry-check'
out.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(root/'code'),str(root/'code/src')]
from scripts import generate_paper_figures as g
p=g.manuscript_parameters()['geometry']
result={}
for kind,recon,point in [('evaporation',g.build_large_p_reconstruction(p),(16.,18.)),
                         ('formation',g.build_formation_reconstruction(p),(-11.,-9.))]:
    u,v,_,_=g.wavepacket_start_indices(recon['tgrid'],*point)
    R,L=g.brl.simulate_wavepacket(recon['gates'],2,recon['NT'],start_u=u,start_v=v,
        initial_state=np.array([1.,0.],dtype=complex),is_left=False)
    intensity,z,zf,tf=g.wavepacket_observable_grids(R,L,recon)
    np.savez_compressed(out/f'{kind}-wavepacket.npz',intensity=intensity,Z=z,depth=zf,time=tf)
    opposite=(intensity-z)/2
    result[kind]=dict(NT=recon['NT'],Z_min=float(np.nanmin(z)),Z_max=float(np.nanmax(z)),
        opposite_component_max_intensity=float(np.nanmax(opposite)),
        total_link_intensity_max=float(np.nanmax(intensity)),
        note='Link-local component weights; not an integrated boundary transmission probability.')
    if kind=='formation':
        g.plot_gate_color_map(recon,p,out,'fig3c_formation_gate_color_map.png',
            title='Formation and evaporation gate transfer strength',draw_formation_lines=True)
        old=np.array(Image.open(g.PAPER_FIG_DIR/'fig3c_formation_gate_color_map.png').convert('RGB'))
        new=np.array(Image.open(out/'fig3c_formation_gate_color_map.png').convert('RGB'))
        result['formation_panel_direct_replot']=dict(old_shape=list(old.shape),new_shape=list(new.shape),
            identical=bool(np.array_equal(old,new)))
        if old.shape==new.shape:
            result['formation_panel_direct_replot']['mean_abs_pixel_difference']=float(np.abs(old.astype(float)-new.astype(float)).mean())
            result['formation_panel_direct_replot']['changed_pixel_fraction']=float(np.any(old!=new,axis=2).mean())
(out/'geometry-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
