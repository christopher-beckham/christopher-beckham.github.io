---
title: Towards a more sane Mac OS user experience, and I am late to the party
layout: default_latex
description: My journey in trying to craft a better user experience for myself on Mac OS X.
---

<a id="org5e4dbd9"></a>

# Towards a more sane Mac OS user experience, and I am late to the party

Despite making the transition to Mac OS from Windows as early as 2014, I don't feel like I have made a particularly strong effort to become a 'power user' in the past eight years. By 'power user', I mean the kind of person that tweaks every little setting and configuration file and hotkey so that they have an optimised workflow that works for them. While I certainly regret not delving deeper into this (and I have my reasons), I figured that this is the year that I try and at least make an effort.

Mac OS is pretty decent out of the box. And I certainly would never go back to Windows for my work life and development-related things. However, over time the user experience gradually started to irritate me, and it's hard to describe that in precise detail because I feel as if it's a culmination of many different things of a relatively minor nature, rather than one or two blaring issues that really stick out. For instance, I recall Windows generally having a 'snappier' and more responsive UI, and I found it weird that the OS didn't even ship with an option to let you snap windows to the edges of the screen; that has to be done with a third party app. Sometimes when I installed a new application, Spotlight had a hard time trying to find it, so instead of simply typing `cmd+space <application name>` I had to instead open Finder, navigate to `/Applications` and actually launch the application from there. Oh, speaking of Finder, I absolutely hate its search with a passion. Many times I had to spin up a terminal window and use `ls` and `grep` instead because it was much more reliable. Ughhhhh. When Apple introduced their [split screen mode](https://support.apple.com/en-us/HT204948), it only let you split the screen with two applications and only on the horizontal axis. Why so feature incomplete??? Eventually I migrated to a decent tiling window manager called [Amethyst](https://github.com/ianyh/Amethyst). Tiling window managers automatically re-arrange your windows, though they can also be sometimes annoying. I'll get to that at the end of this post.

Some months ago I went through the mental crusade of trying to identify what would make my user experience a whole lot better. I concluded that the thing I wanted the most for this OS was to have something like Spotlight where you can search **all open applications** so that you could easily switch to them by invoking a hotkey and typing some text. As it stands, Spotlight doesn't have this feature (though it does let you search through other boatloads of crap), and the only 'power user' way to flick through apps is to use mission control with a swipe gesture or use `cmd+tab` (which is basically `alt+tab` on Windows). This feature is so nice that I also have it installed in my web browser, and it's an absolute time-saver when you have so many tabs open:

<div id="images">
<br />
<figure>
<img class="figg" src="/assets/04/waterfox.gif" alt="" width=600 />
</figure>
<figcaption><b>Figure 1:</b> 
Showing off the Vim Vixen plugin for Firefox. This adds some nice Vim keybindings to let you power through your tabs. Here, I simply press 'b' and then use 'tab' to cycle through all of my open tabs, or even search by typing in a string.
</figcaption>
<br />
</div>

I knew that Linux had such apps available ([dmenu](https://tools.suckless.org/dmenu/), [rofi](https://github.com/davatorium/rofi)) and so I was torn between staying on Mac or transitioning to Linux. When I was experimenting with that transition, I decided to do it using a virtual machine (I recommend [Parallels](https://www.parallels.com/products/desktop/)) just so that I would have the best of both worlds in front of me; after all, if the virtual machine isn't sluggish then I really see no reason why you would want to go through the hassle of dual booting. For a few weeks I played with an Ubuntu distro with i3 as my window manager and it felt pretty decent, apart from how absolutely painful it was to have to trudge through Stackoverflow for every little confusion I had with every little config file I had to modify. It's basically config files galore, each one having its own esoteric syntax you have to learn and with no GUIs in sight to ease the transition. I also had to deal with the fact that I had to be careful that any keybindings I set up for the guest OS (Linux) did not interfere with the host OS (Mac). Ultimately, I didn't feel super comfortable making the transition to my virtual Linux OS because the font rendering was so much better on Mac, not to mention the amazing trackpad experience which I didn't want to give up. Also, if I did go the dual boot route, I didn't want to risk succumbing to this sort of fate (context: a [related thread on Hackernews](https://news.ycombinator.com/item?id=29744419)):

> Well, if you're very opinionated regarding your setup, trying to force macOS into your ways won't work, macOS is great and very easy to use and gives you zero problems but you have to adapt to it. I've also moved after many years of linux and I could not be happier. I like easy and I like to focus on getting my actual work done, I got tired of spending weeks personalizing stuff, dealing with drivers issues, tuning the trackpad, adjusting applications to work with different dpi screens, etc, etc. For me it was a never ending war and a lot of time wasted.

For now, I feel like I have some made some headway with finding something that works for me on Mac. The first is a very awesome tiling window manager called [Yabai](https://github.com/koekeishiya/yabai). Yabai requires [some hacks](https://github.com/koekeishiya/yabai/wiki/Disabling-System-Integrity-Protection) in order to leverage all of its features, but it offers really awesome power-user stuff if you combine it with a hotkey daemon like [skhd](https://github.com/koekeishiya/skhd). Some of its features include:

-   being able to instantly switch workspaces without the annoying swipe animation (with hotkeys);
-   the ability to switch focus on windows in a workspace;
-   being able to swap the positions of windows in a workspace;
-   being able to resize windows with hotkeys;
-   adding rules to control which windows should be tiled and what should remain floating;
-   the ability to set window opacity;
-   &#x2026;and loads more.

It's not all roses, however. Some of these features require you to disable [System Integrity Protection](https://github.com/koekeishiya/yabai/wiki/Disabling-System-Integrity-Protection). Right now I am on a work laptop, and that is definitely something I would not be allowed to do (or even achieve since I don't have admin rights). That ends up reducing the appeal of Yabai, and in that case it may not have much more to offer than any other tiling window manager that lets you customise hotkeys.

At least for my personal laptop, `skhd` has allowed me to assign hotkeys to other yabai functions. Here is me switching workspaces with `cmd+alt+<number>` (the number corresponds to the workspace number in mission control):

<div id="images">
<br />
<figure>
<img class="figg" src="/assets/04/workspace.gif" alt="" width=600 />
</figure>
<figcaption><b>Figure 2:</b>
Cycling through workspaces with my hotkeys. Note that we don't have to endure a swiping animation either, so it's snappy.
</figcaption>
<br />
</div>

These are the commands I have in my shkdrc file to make that work:

```
# focus on a specific workspace
cmd + alt - 1 : yabai -m space --focus 1
cmd + alt - 2 : yabai -m space --focus 2
cmd + alt - 3 : yabai -m space --focus 3
```

If you install [choosem](https://github.com/Granitosaurus/choosem) and combine it with `skhd`, you can basically get the rofi-style application search that I mentioned earlier. For instance, in my `~/skhdrc` file I have the following:

```
shift + cmd - space : /Users/beckhamc/miniconda3/bin/choosem yabai focus
```

(The command `choosem yabai focus` actually makes use of yabai since its API includes a command that lets you get metadata on all of the open windows. That command in particular is `yabai -m query --windows`)

I have this functionality bound to `cmd+shift+space`. (`cmd+space` is still bound to Spotlight, which can still be useful for launching applications.)

<div id="images">
<br />
<figure>
<img class="figg" src="/assets/04/choosem.gif" alt="" width=600 />
</figure>
<figcaption><b>Figure 3:</b>
This is like Spotlight, but I can search open applications and focus to them. It is absolutely crazy that Mac does not have this built into the OS.
</figcaption>
<br />
</div>

Tiling window managers are great, but it can be annoying when they try and tile applications with small windows. Here are some applications I blacklisted in my Yabai config:

```
# use this: yabai -m query --windows
# to help with filtering window types
yabai -m rule --add title=' Preferences$' manage=off
yabai -m rule --add app='Finder' manage=off
yabai -m rule --add app='System Information'  manage=off
yabai -m rule --add app='TV'  manage=off
yabai -m rule --add app='choose' manage=off # float choosem window
yabai -m rule --add app='Emacs' manage=off # emacs has a quirky ux on mac
```

That is all for now.