# 🧸 Antigravity SDK Python (ELI5 - Explained for a 5-Year-Old)

## What does this project do?
Imagine you have many smart toy robots (we call them **agents**). These robots all want to build a giant Lego castle together. To make sure they don't get in each other's way and can talk to one another, they need some help.

This project acts like a **magical, invisible road network** (a *mesh network*):
1. **Who does what?** It draws a map showing exactly which robot is friends with which tool (this is the *dependency graph*).
2. **No accidents:** It makes sure that two robots do not try to go through the same tiny door at the same time (we call this *port leasing*).
3. **Sharing smart ideas:** When a robot has a great idea, this system whispers it super fast into the ears of all the other robots (this is *vector search* and *WebAssembly execution*).

## Why does it exist?
So our smart robots can work together perfectly as a team without arguing or bumping into each other!
