from mx2_twisted_bilayer import MX2
mx2 = MX2()
mx2.to_bilayer('2H')
mx2.add_hopping(spin=1, dist_cut=10)
