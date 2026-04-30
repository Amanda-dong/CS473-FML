# Streamlit UI Change Plan

Updated: 2026-04-30

## Goal

Improve the Streamlit app so it is more user friendly, reduces redundant UI elements, moves methodology off the front page and into sidebar navigation, adds clear page-level summaries, and introduces an NLP-based concept entry path alongside the existing toggle-based recommendation controls.

This document is a working checklist and can be updated as tasks are completed.

## Scope Summary

The current app has:

- a main home page in [frontend/app.py](/Users/Catherine/Desktop/CS473-FML/frontend/app.py)
- a front-page Methodology tab
- a Data Sources tab
- a separate review-zone demo page in [frontend/pages/2_Review_Zones_Demo.py](/Users/Catherine/Desktop/CS473-FML/frontend/pages/2_Review_Zones_Demo.py)
- filter and scenario controls distributed across sidebar components

The requested UI changes are:

- move Methodology off the main front page and make it accessible from sidebar page navigation instead
- add a short, informative overview at the top of every page
- identify and remove redundant UI features
- add an NLP concept-description flow alongside the existing toggle-based concept selection flow
- keep both concept entry modes available so users can choose either one

## Task Checklist

### 1. Discovery and Current-State Audit

- [x] Review the current Streamlit page structure and list all pages currently exposed to users.
- [x] Document which controls live in the sidebar versus the main content area.
- [x] Identify all areas where the same user intent is captured more than once.
- [x] Identify any tabs, buttons, panels, or filters that create confusion or duplication.
- [x] Confirm which pages need user-facing overview text at the top.

### 2. Page Navigation Restructure

- [x] Remove the Methodology tab from the main front page tab layout in [frontend/app.py](/Users/Catherine/Desktop/CS473-FML/frontend/app.py).
- [x] Convert Methodology into a standalone Streamlit page reachable from the sidebar page menu.
- [x] Verify that Methodology still renders correctly after being moved out of the front-page tab flow.
- [x] Confirm that the main page navigation order is intuitive after the Methodology move.
- [x] Review whether Data Sources should remain on the main page or also become a standalone page.
  Decision: keep Data Sources on the main page for this pass.

### 3. Page Intro Summaries

- [x] Add a short summary block at the top of the main recommendations page explaining what the page does and how to use it.
- [x] Add a short summary block at the top of the Methodology page explaining what users will learn there.
- [x] Add a short summary block at the top of the Data Sources area explaining what data freshness and source information means.
- [x] Add a short summary block at the top of the review/NLP-related page(s) explaining their purpose and outputs.
- [x] Make sure each page summary is brief, informative, and non-technical enough for first-time users.
- [x] Ensure the summary text does not crowd the page or push important controls too far down.
- [x] Rewrite each page intro so it speaks directly to the user as a halal restaurant merchant, not as a generic analyst or demo user.
- [x] Make each page intro more action-oriented by telling the merchant what to click first and what they will get from the page.
- [x] Add lightweight wayfinding language so each intro helps the merchant decide whether they are in the right place or should move to another page.
- [x] Review page intros for tone so they feel helpful, confident, and product-like rather than academic or internal.
- [x] Add very brief descriptions near key page features so merchants understand what each chart, filter, card, table, or map section does.
- [x] Keep feature descriptions short enough that they guide the user without making the page feel crowded.

### 4. Redundant Feature Cleanup

- [x] Audit recommendation controls for overlap between the concept selector, input form, and scenario panel.
- [x] Decide which duplicated controls should be removed, merged, or relabeled.
- [x] Remove redundant user inputs that create the same effect in multiple places.
- [x] Simplify the top-level recommendations view so the primary user path is obvious.
- [x] Review whether any repeated charts, tables, or download actions should be combined or relocated.
- [x] Review page titles, headers, and section labels for repeated or unnecessary wording.

### 5. Concept Input Experience: Toggle Path and NLP Path

- [x] Define the two concept-entry modes clearly:
  - existing structured/toggle-based path
  - new NLP free-text path
- [x] Decide where users choose between the two modes in the UI.
- [x] Add a mode switch that makes it obvious users can either describe their restaurant idea in text or use structured controls.
- [x] Preserve the current toggle-based concept workflow so existing functionality is not lost.
- [x] Design the NLP input experience so a user can describe the restaurant concept they want to create in plain language.
- [x] Define how NLP-derived concept interpretation maps onto the existing recommendation request schema.
- [x] Decide what happens when NLP output is incomplete, ambiguous, or conflicts with existing toggles.
- [x] Decide whether the NLP path should pre-fill structured fields, run independently, or allow user review before execution.
  Decision: the NLP-style path runs independently for now and shows the parsed concept back to the user before recommendations run.
- [x] Add user guidance so the difference between the two input modes is easy to understand.

### 6. NLP UI Requirements

- [x] Create a dedicated UI section for NLP concept entry.
- [ ] Add helper text with examples of good restaurant concept prompts.
- [x] Decide whether NLP should parse cuisine, price tier, target customer, health positioning, and risk tolerance.
- [ ] Define loading, success, and error states for NLP interpretation.
- [x] Decide how parsed NLP results are displayed back to the user before recommendations run.
- [ ] Add a clear fallback path so users can switch back to the structured controls if NLP parsing is not satisfactory.
- [ ] Decide whether the review-zones demo page should stay separate or evolve into part of a broader NLP experience.
- [x] Define exactly which NLP outputs should populate the existing request fields:
  - `concept_subtype`
  - `price_tier`
  - `risk_tolerance`
  - any future merchant intent fields
- [x] Add a confirmation step so merchants can review and edit the NLP-parsed concept before running recommendations.
- [ ] Decide whether NLP should support prompts about launch timing, customer type, service style, and neighborhood goals in addition to cuisine.
- [ ] Add tasks for handling vague prompts, multi-concept prompts, and prompts that are not clearly halal-focused.
- [ ] Add evaluation tasks for checking whether NLP parsing is mapping merchant descriptions into the correct halal concept category.
- [ ] Plan how to display the merchant's original description alongside the parsed concept so the recommendation flow remains transparent.
- [ ] Add a task to connect NLP-derived concept parsing to the existing survivability workflow from phase two.
- [ ] Decide whether NLP should only choose a concept subtype or also influence survivability-related inputs such as merchant viability framing, risk tolerance defaults, and launch-context assumptions.

### 7. Main Page UX Improvements

- [x] Rework the main page layout so the user journey is clearer from top to bottom.
- [x] Ensure users understand where to start when they first open the app.
- [x] Clarify the difference between search filters, concept definition, and scenario comparison.
- [ ] Review whether the current Top Picks, map, results cards, and download actions appear in the best order.
- [ ] Decide whether summary metrics at the top of the page are still useful or should be revised.
- [ ] Review spacing, headings, and divider usage for readability.
- [x] Add short helper text for the main page features so the merchant knows what each section is for before interacting with it.
- [x] Reduce visual clutter by removing or regrouping sections that make the page feel busy or harder to scan.
- [x] Make the most important actions more obvious so the merchant can quickly move from concept entry to shortlist review.
- [x] Add a featured “Top Match” section so merchants can immediately see the single strongest recommendation for their query.
- [x] Decide how the Top Match should be highlighted:
  - featured card
  - pinned first recommendation
  - highlighted map state
  - summary hero panel
- [x] Add a short explanation showing why the Top Match is the strongest current fit for the merchant’s halal concept.
- [x] Make sure the Top Match remains visually clear even when the user turns on comparison mode or changes filters.

### 8. Content and Labeling Cleanup

- [ ] Rewrite confusing labels into more user-friendly language.
- [ ] Standardize wording for concept, recommendation, zone, risk, and confidence terms.
- [ ] Make the UI sound more like a product for restaurant users and less like an internal demo.
- [ ] Review button labels, section headers, captions, and helper text for consistency.
- [ ] Ensure NLP and structured-input terminology is consistent across the app.
- [x] Update top-of-page summaries so they guide merchants toward the next best action instead of only describing the page.
- [x] Make sure summary copy assumes the user is trying to open a halal restaurant and wants help choosing where and when to launch.
- [x] Add concise feature-level labels or captions wherever the current UI assumes the user already knows what a control or section does.
- [x] Simplify copy and layout together so the app feels cleaner, easier to scan, and more user friendly overall.

### 9. Technical Integration Tasks

- [ ] Update page imports and routing to match the new navigation structure.
- [ ] Refactor any shared page-intro components if repeated summary sections are implemented in multiple places.
- [ ] Refactor any sidebar logic needed to support multiple concept-entry modes cleanly.
- [ ] Confirm that session state behaves correctly when switching between NLP and toggle modes.
- [ ] Confirm that reset behavior works correctly for both concept-entry modes.
- [ ] Confirm that comparison mode still works after any input-flow changes.
- [ ] Define the backend path from NLP-parsed concept input into the existing recommendation request object.
- [ ] Define the backend path from recommendation concept input into the existing phase-two survivability model or its heuristic fallback.
- [ ] Decide whether the UI should rely on the existing API response fields like `survival_risk` or call a survivability-specific path separately for NLP-enhanced experiences.
- [ ] Add a task to verify that the Top Match highlight always stays in sync with recommendation sorting, filters, comparison mode, and exports.

### 10. QA and Validation

- [ ] Verify the Methodology page is no longer shown as a tab on the front page.
- [ ] Verify the Methodology page is accessible from sidebar navigation.
- [ ] Verify every user-facing page begins with a short overview/summary section.
- [ ] Verify redundant features actually removed do not break any user path.
- [ ] Verify users can successfully run recommendations using only structured controls.
- [ ] Verify users can successfully run recommendations using only NLP concept input.
- [ ] Verify users can switch between NLP and structured modes without confusing state carryover.
- [ ] Verify layout works at common laptop screen widths.
- [ ] Verify layout remains readable on narrower screens.
- [ ] Verify no existing core recommendation workflow regresses.
- [ ] Verify the Top Match highlight always matches the highest-ranked recommendation after filters are applied.
- [ ] Verify the NLP path still produces stable survivability and recommendation outputs after concept parsing.
- [ ] Verify merchants can understand why the Top Match is highlighted without needing to interpret technical model language.

### 11. Ship Readiness

- [ ] Do a final copy pass for grammar, consistency, and tone.
- [ ] Confirm all new UI text is helpful but concise.
- [ ] Confirm the app still launches cleanly with `streamlit run frontend/app.py`.
- [ ] Confirm no unused or dead UI code remains after cleanup.
- [ ] Update any documentation that references the old front-page Methodology tab.
- [ ] Capture before/after screenshots or notes for the final review.

## Open Design Decisions

- [ ] Should Data Sources remain a tab on the main page or move into sidebar page navigation too?
- [ ] Should NLP concept entry live on the main page or on a dedicated page/section?
- [ ] Should NLP parsing automatically run recommendations or ask the user to confirm the parsed concept first?
- [ ] Should NLP mode pre-populate the structured controls for transparency and editing?
- [ ] Should the review-zones demo remain a separate experimental page or be folded into the new NLP experience?

## Deliverables for This UI Work

- [ ] Main page no longer includes Methodology as a front-page tab.
- [ ] Methodology is available through Streamlit page navigation in the sidebar.
- [ ] Every page has a concise top-of-page summary.
- [ ] Redundant UI features are removed or merged.
- [ ] Users can define a restaurant concept through either:
  - structured/toggle controls
  - NLP free-text description
- [ ] The updated UI is ready for final QA and release.

## Notes

- This file is planning only.
- No UI implementation work has started yet.
- As tasks are completed, update the checkboxes and add any new sub-tasks discovered during implementation.
