# Reliability Checklist

## Queue invariants

- Only one worker owns a live lease.
- Every claim increments `attempts`.
- An expired lease can be reclaimed.
- A stale lease cannot mutate job state.
- Completion and effect insertion are atomic.
- Retry delays follow an injected clock.
- A dead job can be explicitly requeued.

## Failure injection

- Kill a worker after claiming but before completion.
- Let two workers race for the same pending row.
- Reuse an old lease after a newer worker claims the job.
- Retry complete after the effect already exists.
- Upgrade an old database twice.
- Send termination while a worker is processing.

## Evidence quality

- Tests assert database state, not only return values.
- Each test has an observable failure mode.
- Hidden checks cover behaviors beyond the visible examples.
- A no-op or biased implementation receives a clearly lower score.

