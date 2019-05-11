# Cosmological Simulations

### Compres multiple Cholla Files to a single file

```
mpirun -n N python tools/compress_all_data.py SNAPSHOTS_DIR OUTPUT_DIR
```

This will take the cholla output locaed in **SNAPSHOTS_DIR** and write the single files in **SNAPSHOTS_DIR/OUTPUT_DIR**