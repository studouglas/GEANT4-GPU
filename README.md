# Porting GEANT-4 Particle Simulation to GPU

GEANT-4 is a widely-used simulation program used to simulate particle
interactions. There are currently several members of McMaster's Engineering
Physics department that use the program, and are being limited by the
performance of the software. This means that they cannot simulate particle
interactions that take place over the course of minutes (or even seconds), and
they also can't simulate large numbers of particles.

Increasing the runtime of the simulation or the number of particles would
greatly increase the accuracy of their results, allowing the researchers to
understand the systems they're modeling better. This is especially true when
modeling complex systems, such as McMaster's nuclear reactor. Depending upon the
level of success of the project, the solution could potentially benefit groups
that use GEANT-4 outside of McMaster as well.
