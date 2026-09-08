# BSI QuakeML label provenance audit

Parsed from downloaded official QML zips. **No model scores on BSI waveforms.**

| ZIP | events | S picks | S manual | S auto | unique S wfIDs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20210101_20210430__1__INGV__QML.zip | 5356 | 103048 | 89706 | 13342 | 1986 |
| 20210501_20210831__1__INGV__QML.zip | 6033 | 124457 | 102662 | 21795 | 2036 |
| 20210901_20211231__1__INGV__QML.zip | 5144 | 104131 | 86627 | 17504 | 2025 |
| 20220101_20220430__1__INGV__QML.zip | 5065 | 102846 | 87241 | 15605 | 2071 |
| 20220501_20220831__1__INGV__QML.zip | 6082 | 123325 | 101356 | 21969 | 2227 |
| 20220901_20221231__1__INGV__QML.zip | 6455 | 136325 | 113939 | 22386 | 2149 |

## Auditable filter (draft)

1. Keep `phaseHint` starting with `S`.  
2. Prefer `evaluationMode == manual` for the primary independent label set.  
3. Keep `automatic` S in a separate bucket (not mixed into primary labels).  
4. Require waveformID network+station and parseable pick time.  
5. Bulletin headline S counts ≠ usable three-component traces after channel/window filters.  
6. Do **not** treat every S pick as manual without checking `evaluationMode`.

Full JSON: `artifacts/results/paper_strengthening_v1/bsi_catalog/quakeml_label_audit.json`
