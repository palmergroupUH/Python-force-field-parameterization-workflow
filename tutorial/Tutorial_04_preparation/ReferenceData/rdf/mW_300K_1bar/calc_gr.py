import objective.rdf_matching.calc_pair_correlation


calc_gr = objective.rdf_matching.calc_pair_correlation.RadialDistribution("traj.dcd",2,10,100,5000)

calc_gr.compute() 

calc_gr.dump_gr("Ref.gr")

