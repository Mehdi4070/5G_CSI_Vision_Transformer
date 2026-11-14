# Présentation de soutenance

## Fichiers

- `presentation.tex` : diapositives Beamer 16:9 ;
- `presentation.pdf` : présentation compilée ;
- `speaker_script.md` : script oral détaillé.

Avant la soutenance, remplacer les macros `Auteur` et `Encadrant` au début de
`presentation.tex`.

## Compilation

Depuis ce dossier :

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error presentation.tex
```

Le diaporama utilise directement les figures produites par les expériences du
projet. Il faut donc conserver la structure des dossiers du dépôt.
