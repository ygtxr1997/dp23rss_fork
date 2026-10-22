"""Convert official pi0.5 JAX assets into the standalone dense model layout."""
import argparse
from diffusion_policy.model.pi05.checkpoint import convert_checkpoint


def main():
    """Run one explicit conversion without downloading or overwriting assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-model', required=True)
    parser.add_argument('--output-model', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--norm-stats')
    parser.add_argument('--source-uri', required=True)
    args = parser.parse_args()
    print(convert_checkpoint(args.input_model, args.output_model, args.tokenizer,
                             norm_stats_path=args.norm_stats, source_uri=args.source_uri))


if __name__ == '__main__':
    main()
