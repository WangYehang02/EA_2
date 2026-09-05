# Discussion Points (evidence-supported only)

1. **Candidate generation vs candidate selection**  
   Recoverable headroom often already sits inside UNION; the studied failure mode is mis-ranking near the top, not inventing a new pick.

2. **Why rank-2 near-miss is common**  
   Forensic: 83.8% of recoverable good candidates are rank 2; top-3 covers 97.9%. Pairwise c1/c2 is matched to this structure.

3. **Usefulness of train-only propagation residual**  
   Residual control improves over fixed and is nearly equivalent (r≈0.998) to the same train-only travel-time geometry—useful prior, not new external information.

4. **Why pairwise ranking may outperform absolute scoring**  
   Confirm: scalar exceeds residual control (+0.0076, CI excludes 0). Relative comparison of c1 vs c2 can capture local ambiguity beyond a single absolute residual kernel.

5. **Why waveform conditional increment is small**  
   Waveform-only helps vs fixed in isolation, but waveform+scalar vs scalar is immaterial (CI crosses 0). Correct claim: negligible conditional improvement once scalar ranking features exist.

6. **Why multi-station geometry did not dominate**  
   Moveout legal gain ≤ / ≈ single-station resid control; predictions agree ≈98.3%. Apparent multi-station benefit is largely single-station residual driven.

7. **Catalog-assisted task scope**  
   Method uses current-event source/origin metadata. Claims must stay inside catalog-assisted reranking, not blind picking.

8. **Limitations**  
   - Requires frozen candidate set  
   - Uses catalog-assisted source/origin information  
   - Does not solve candidate-missing cases  
   - Not blind continuous detection  
   - Current evaluation dataset/domain only  

9. **Future work (not claimed as proven)**  
   - Blind / estimated-source variant  
   - Better candidate generation for missing cases  
   - External regional validation  

Do not present future work as already effective.
