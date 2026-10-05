### Docs: the Microduck bundle section switches to a key its own sketch holds

`docs/learn/policies/microduck.md` said `bundle.switch("alpha_stand")` while the
sketch below it keys the bundle `{"walk", "stand"}`, so following both raised
`ValueError: unknown skill 'alpha_stand'`. The prose now says
`bundle.switch("stand")`, and a page check refuses any key the section names that
its sketch does not hold.
