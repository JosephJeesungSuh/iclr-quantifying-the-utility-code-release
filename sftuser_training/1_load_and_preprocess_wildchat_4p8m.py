"""Appendix C.2: the same pipeline, excluding all WildChat-1M conversation hashes."""
from preprocessing import main

if __name__ == '__main__':
    main(larger_corpus=True)
