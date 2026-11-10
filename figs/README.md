# Figures / Figuras

- `*.png`: English figures used by `README.md`.
- `*-pt_BR.png`: figuras em português usadas por `README-pt_BR.md`.

O comando abaixo gera as figuras em português e atualiza o artigo em português:

```bash
.micromamba/micromamba run -p .micromamba/envs/hhc python -m src.report
```

As figuras em inglês são preservadas. Exportações opcionais em PDF seguem o mesmo sufixo.
