# Suggested Follow-up Tasks

The review identified several issues spanning typos, bugs, documentation inconsistencies, and testing gaps. Each item below is framed as a concrete task with a short justification and the relevant references.

## Typo Fix
- **Normalize Arabic spelling for "إرشاد" in the keyword list.** The current entry in `AR_KEYWORDS` omits the hamza ("ارشاد"), which is inconsistent with standard Arabic orthography and the rest of the list. Correcting the spelling keeps terminology consistent for downstream matching. *(See `unified_nlp_toolkit.py`, line 636.)*

## Bug Fix
- **Preserve trailing tokens in `group_texts` during dataset preparation.** When the combined length of `input_ids` is shorter than `block_size`, the current implementation truncates everything to zero tokens, yielding empty datasets and causing training to fail on small corpora. Update the logic to retain the remainder (e.g., by returning the final partial block or skipping truncation when the dataset is smaller than `block_size`). *(See `unified_nlp_toolkit.py`, lines 362-374.)*

## Documentation Alignment
- **Expand the README to document the CLI features.** The repository README only contains the project title, while the script exposes extensive export, training, generation, and serving functionality. Align the README with the script docstring and CLI so that usage instructions are discoverable. *(See `README.md`, line 1, and the module docstring in `unified_nlp_toolkit.py`, lines 1-3.)*

## Test Improvement
- **Add unit tests for the `is_relevant` filter.** The relevance gate is crucial for the Gradio counseling interface, yet it currently lacks regression tests. Introduce coverage that exercises Arabic and English prompts, as well as negative cases, to guard against inadvertent keyword or boundary regressions. *(See `unified_nlp_toolkit.py`, lines 604-632.)*

