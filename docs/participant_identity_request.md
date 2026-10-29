# Request to verify ZuCo cross-release participant overlap

Prepared 2026-10-29. The user subsequently showed the inquiry in Gmail's Sent
folder. No response has been provided in this session. The assistant did not
send the email. The user initially chose to seek identity evidence, then chose
to continue without waiting, using diagnostics and explicitly exploratory
transfer with participant overlap unknown (see [continued work](continue_work.md)).
The original message draft follows
for the project record; it does not need to be sent again.

The supplied ZuCo2 paper (`1912.00903v3.pdf`, first page) lists Nora Hollenstein
at **noraho@inf.ethz.ch**. This is the paper-listed address, not a newly verified
current affiliation/contact. The [official ZuCo2 OSF project](https://osf.io/2urht/)
also identifies the project contributors and can be used to find current contact
details if needed.

## Email draft

**Subject:** ZuCo 1.0/2.0 participant overlap for cross-dataset EEG evaluation

Dear Dr. Hollenstein,

I am working on a university project evaluating EEG-to-sentence retrieval using
the normal-reading data from ZuCo 1.0 and ZuCo 2.0. I would like to train on one
release and evaluate on the other while ensuring no participant occurs in both
the training and test cohorts.

Could you please confirm whether any of the 12 released ZuCo 1.0 participants
also participated among the 18 released ZuCo 2.0 participants?

If the cohorts are disjoint, is there a publication, dataset note or other
reference we can cite? If there is overlap, is an anonymized cross-release
subject-ID mapping available, or could you advise which subject IDs should be
excluded to obtain disjoint cohorts? No personal identities are needed.

We have reviewed the ZuCo 2.0 paper, particularly Section 3.1, and the OSF
documentation. We found the intentional sentence overlap but could not establish
participant overlap from those sources. The different Z/Y label prefixes alone
do not establish that the participants differ.

Thank you for your time and for making these datasets available.

Best regards,
Ziad

## Cohort identifiers, if requested

- ZuCo1: ZAB, ZDM, ZDN, ZGW, ZJM, ZJN, ZJS, ZKB, ZKH, ZKW, ZMG, ZPH.
- ZuCo2: YAC, YAG, YAK, YDG, YDR, YFR, YFS, YHS, YIS, YLS, YMD, YMS, YRH,
  YRK, YRP, YSD, YSL, YTL.

## Handling a reply

Keep the reply/source, its date and its stated cohort scope with the experiment
records. An explicit confirmation of disjoint released cohorts can support
version-scoped participant IDs, with the source recorded. An overlap mapping
must assign the same person ID to linked source labels and remove overlapping
participants before strict transfer. Recompute cohort manifests, train-only
normalization, text-cache coverage and model artifacts after exclusions as needed.
Statements about repeated stimuli, new recordings, different file labels or a
different recording year do not by themselves settle participant identity.

Do not publish private correspondence or personal information without the
sender's permission; a public source or permission to cite the specific cohort
confirmation is preferable for the final report. Until evidence is received,
strict transfer remains unexecuted and no cross-release identities are invented.
