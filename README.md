# cpx

**Note:** Currently, **cpx** is a rough draft of an initial CPU benchmark for a
set of base utilities that enable memory efficient and fast processing of calcium
imaging data. 

Most source extraction libraries (that I'm aware of) have trouble scaling past
~50 GBs of data, are not designed with memory efficiency in mind, and
additionally leave a lot of optimizations and flexibility on the table. Early
CPU benchmarks lead me to believe that a ~5-10x increase in performance (with 
better memory management) is possible on CPUs alone, while maintaining similar
(if not exact) source extraction results. Additionally, GPUs *might* lead to
another order of magnitude increase in performance. 

The goal of **cpx** in the long run, would be to have a set of battle tested 
utilities (for filters, metrics, registration, extraction, etc.) that are
designed from the bottom up with memory efficiency and performance in mind (on
both CPUs and GPUs). This also extends to 2-photon stacks. 

If your interested in this problem and would like to talk (either as an end-user
or developer), send me an email at RyanIRL (at) icloud (dot) com. 


## Installation 

```
pip install cpx
```

## License

MIT, see `LICENSE.txt`.


