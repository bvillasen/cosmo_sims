# Cosmological Simulations

### Compres multiple Cholla Files to a single snapshot file

```
mpirun -n N python tools/compress_all_data.py SNAPSHOTS_DIR OUTPUT_DIR
```

This will take the cholla outputs located in **SNAPSHOTS_DIR** and write the single files in **SNAPSHOTS_DIR/OUTPUT_DIR**. The number of mpi processes **N** is the number of snapshots that will be compressed.

### Zeldovich Pancake analysis

```
mpirun -n N python analysis/zeldovich.py INPUT_DIR OUTPUT_DIR
```

This will take the single file snapshots located in **INPUT_DIR** and save the images in **OUTPUT_DIR**. The number of mpi processes **N** is the number of snapshots that will be analyzed.
