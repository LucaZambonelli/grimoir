#!/usr/bin/env python
# -*- coding: utf-8 -*-

# RUN CHECK FOR GRIMOIR PACKAGE

import os

def cls():
    os.system('cls' if os.name=='nt' else 'clear')

cls()
prompt = input(
    "Which check do you want to perform?"
    + "\n1 - one degree of freedom kinematics"
    + "\n2 - two degrees of freedom: open dyad, direct"
    + "\n3 - two degrees of freedom: open dyad, inverse"
    + "\n4 - two degrees of freedom: five bar link"
    + "\n\n"
)
cls()

if prompt == "1":
    import onedof
elif prompt == "2":
    import directdyad
else:
    print("puppa!")
