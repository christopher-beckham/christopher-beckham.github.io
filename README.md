# Installation

This Ruby stuff can be frustrating, and I don't write Ruby. As of time of writing, I had to basically install a specific version of Ruby via Brew, as well as Webrick.

```
brew install ruby@3.1
bundle add webrick
```

## Using emacs to publish

Here is an excerpt from my `settings.org` file:

```
  (setq org-publish-project-alist
        '(
          ;; My own Github repo for tinkering org-to-latex
          ("org-to-latex-test"
           :base-directory "~/github/org-to-latex"
           :base-extension "org"

           :publishing-directory "~/github/org-to-latex/output"
           ;; :publishing-function org-html-publish-to-html

           ;; This exports from org to latex to pdf but I get compiler
           ;; errors complaining about a lack of \begin{document} ... \end{document}.
           :publishing-function org-latex-publish-to-pdf
           ;; :html-extension "html"
           :body-only t)

          ;; My website blog. Write blogs in org to start off with, then automagically
          ;; export them to html for Jekyll.
          ("christopher-beckham.github.io"
           :base-directory "~/github/christopher-beckham.github.io/_org/org-publish"
           :base-extension "org"

           :publishing-directory "~/github/christopher-beckham.github.io/_posts"
           :publishing-function org-html-publish-to-html
           :html-extension "html"
           :body-only t)))
```

Blog posts are now written in `base-directory` and are automatically exported to `_posts`.
