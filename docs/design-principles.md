# ALI Express design principles

**Status:** Living and leading guidance for all user-facing changes.

## Purpose and scope

ALI Express helps primary schools create group arrangements. It must be possible for a
teacher or support coordinator to use the application independently, even when they are
new to ALI Express and have no technical knowledge.

These principles guide UX, UI, user-facing copy, interaction and accessibility across the
whole application. They describe the intended experience and the reasoning behind it.
They do not prescribe exact page copy, component markup, colour values or implementation
details.

## Core principles

### Make mistakes difficult and recovery easy

Design the interface so that the expected action is apparent and invalid input is hard to
enter. Prefer clear choices, sensible defaults, native controls and timely guidance over a
long list of rules shown in advance.

When something does go wrong:

- retain everything the user entered wherever possible;
- say concretely what went wrong;
- say exactly what the user needs to change and where;
- bring the user to the place where the problem can be fixed;
- never require the user to understand an internal error or technical concept.

Potentially destructive or replacing actions state their consequence before they happen.

### Support independent use from the first visit

Every page or state should let a new user understand:

1. what is happening here;
2. what they can or need to do;
3. why the action or information matters;
4. what will happen next.

The normal journey must not depend on training, documentation or knowledge of how the
calculation works internally. Help should be present at the moment it is useful.

### Consistency creates clarity

Consistency is part of making the application understandable. Use the same words,
interaction patterns, visual hierarchy and feedback for the same concepts and actions.
Make genuinely different actions or consequences visibly different.

Reuse an established pattern unless another approach gives the user a clear benefit.
Avoid page-specific novelty for its own sake. A justified exception should still feel like
part of ALI Express.

### Give each state a clear purpose

Use one clear page heading that names the task or outcome. Make the main action easy to
find and give each state one visually dominant next step. Back, cancel, delete and
alternative routes remain visible but are visually secondary.

Empty and nearly empty states still show every sensible way forward. Important names,
choices and actions remain fully readable and predictably positioned.

### Present information in useful layers

Start with the normal task and the information most people need. Explain why something
matters before an action when that explanation affects confidence or the choice being
made.

Place examples, exceptions and technical depth afterwards or in a clearly named
disclosure. Never hide information that is required to complete the task or make an
informed choice in a tooltip or closed disclosure.

### Keep the user informed and in control

Actions have predictable results. The application does not silently discard input or
change the meaning of saved data. Back navigation returns to the expected state.

For longer processes, show that work is continuing, what stage it has reached and whether
the information on screen is still provisional. Make clear when the user can safely leave
and return later. Explain optional choices and the consequences of unusually strict
settings at the point where they matter.

## Writing and voice

### Clear, readable and actionable

User-facing copy is written in plain Dutch. Use familiar words, short sentences and active
verbs. Give each paragraph one job. Remove technical detail that the user does not need to
make a choice or complete a task.

Button labels are short and describe the action or destination. Instructions tell the
user what to do. Error messages are concrete recovery instructions, not descriptions of
the application's internal state.

Address the user with **je/jij**. Use **we/wij** when guiding the user and **ALI Express**
when explaining what the product does.

### Warm, welcoming and enthusiastic

The copy should make the user feel welcome and confident in the application. Write with
warmth and energy, without becoming technical, distant or corporate.

Lead with the benefit and the clear message. Do not dilute a simple promise with technical
detail or a stack of precautionary qualifications. Add nuance where omitting it could
create a wrong expectation or lead to a wrong decision. Accuracy and enthusiasm should
reinforce each other: ALI Express can sound confident about what it genuinely does.

### Precise terminology

Use one term for one concept. Do not replace established terms merely for variety. The
detailed meanings and avoided alternatives in [`CONTEXT.md`](../CONTEXT.md) are the
canonical project vocabulary.

For groups in the user interface:

- normally use **groep**;
- use **groep in de nieuwe indeling** when it must be distinguished from a current group;
- use **nieuwe groep** only for a group that the user actually creates;
- do not expose the internal term **bestemmingsgroep**.

Keep meaningful distinctions intact. A preference is weighed by the calculation. An
exclusion, extra requirement or spreading maximum always applies. Explain that difference
in ordinary language whenever it affects the user's choice.

## Visual character

ALI Express feels warm, cheerful, playful and human. It should not look formal, corporate
or predominantly grey. Its visual character can use energetic orange, generous space,
friendly typography, rounded forms and simple illustrations or icons.

Playful does not mean childish or busy. Decoration supports the task and does not compete
with it. Icons reinforce text rather than replace it. More expressive or celebratory
moments are appropriate when the user reaches a genuine milestone; task-heavy screens can
be calmer while retaining the same character.

Exact fonts, colours, sizes and interaction states belong in the shared styles and design
tokens. This document records their intended effect rather than duplicating their current
values.

## Accessibility and resilience

Accessibility is part of the design from the start:

- prefer semantic HTML and native controls;
- make the complete flow usable with a keyboard and keep focus clearly visible;
- meet at least WCAG AA contrast for text and interactive states;
- never communicate meaning through colour alone;
- preserve all information at 200% zoom;
- prevent horizontal page scrolling at 320 CSS pixels;
- allow long learner, group and arrangement names to wrap without hiding information;
- make hidden content genuinely unavailable until it is shown;
- manage focus when dialogs open and close;
- announce dynamic changes only when they are meaningful;
- respect the user's reduced-motion preference.

Accessibility improvements must preserve the meaning of the product. A copy or layout
change does not silently alter calculation behaviour, weights, saved data or distribution
modes.

## Applying and maintaining these principles

- Add a rule here when it applies across the application or is likely to guide several
  future changes.
- Keep exact copy, page-specific decisions and temporary review status in the relevant
  implementation plan.
- Keep exact visual values and component states in the code that implements them.
- Use an ADR for a lasting architectural decision with meaningful alternatives and
  consequences, not for general design guidance.
- Test lasting user behaviour rather than duplicating complete text passages in tests.
- Review user-facing work in a real browser with a keyboard, long names, narrow viewports
  and zoom, in addition to automated checks.
- Update this document when user feedback reveals a reusable principle. Avoid adding a
  rule for every isolated preference.

Before accepting a user-facing change, ask:

1. Can a first-time, non-technical user understand the task and next action?
2. Is it difficult to make a mistake and straightforward to recover from one?
3. Is the wording clear, actionable, warm and confident?
4. Does it use the established terminology and interaction patterns?
5. Does the user retain their input, overview and control?
6. Does it remain usable with a keyboard, zoom, narrow screens and long names?
