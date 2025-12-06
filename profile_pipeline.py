import cProfile
import cosmosis
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("params_file", type=str, help="Path to the parameters file")
parser.add_argument("output", type=str, help="Output file for the profiling results")
args = parser.parse_args()

pr = cProfile.Profile()
pipeline = cosmosis.LikelihoodPipeline(args.params_file)
p = pipeline.start_vector()
pipeline.run_results(p)


pr.enable()
pipeline.run_results(p)
pr.disable()


pr.dump_stats(args.output)
print(f"Profiling results saved to {args.output}")