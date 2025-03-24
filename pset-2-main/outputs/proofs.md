# Short-Answer Questions

## Part 1.1a
In this problem, we are asked to prove that $a \sin(x) + b\cos(x)$ can be expressed as $c\sin(x+\phi)$ for some appropriate values for $c$ and $\phi$. To see this, note that we can equivalently write 
\[a \sin(x) + b\cos(x) = \sqrt{a^2 + b^2}\left(\frac{a}{\sqrt{a^2 + b^2}}\sin(x) + \frac{b}{\sqrt{a^2 + b^2}}\cos(x)\right).\]

On the other hand, by the sum formula, we have 
\[c\sin(x + \phi) = c\left(\sin(x)\cos(\phi) + \cos(x)\sin(\phi)\right).\]

Matching coefficients, we set $c = \sqrt{a^2 + b^2}$ and $\phi$ such that $\cos(\phi) = \frac{a}{\sqrt{a^2 + b^2}}, \sin(\phi) = \frac{b}{\sqrt{a^2 + b^2}}$. In other words, we would have $\phi = \arctan(\frac{b}{a})$. 




## Part 1.1b
Let $(a, b)$ be on the unit circle, that is, we have $a^2 + b^2 = 1$, we need to prove $y(x) = a\sin(x) + b\cos(x)$ always have unit $L^2$ norm. Using results from Part 1.1a, we already know $y(x)$ can be written as $c\sin(x + \phi)$ for some $c$ such that $c^2 = a^2 + b^2 = 1$. Computing the $L^2$ norm over $[-\pi, \pi]$, we have 
\[\|y\|_{L^2([-\pi,\pi])}^2 = \frac{1}{\pi}\int_{-\pi}^\pi y^2(x)dx = \frac{1}{\pi}\int_{-\pi}^\pi c^2\sin^2(x+\phi)dx = \frac{1}{\pi}\int_{-\pi}^\pi\sin^2(x+\phi)dx.\]


Noting $\sin^2(x)$ has period $\pi$, we have 
\[\frac{1}{\pi}\int_{-\pi}^\pi\sin^2(x+\phi)dx = \frac{1}{\pi}\int_{-\pi}^\pi\sin^2(x)dx = \frac{1}{\pi}\int_{-\pi}^\pi\frac{1 - \cos(2x)}{2}dx = 1.\]

Hence the proof. 






## Part 2.3

What we noticed is that the eigenvalues and eigenfunctions come in pairs (except for the one corresponding to constant eigenfunction). Moreover, the eigenfunctions corresponding to a pair of eigenvalues take the form of linear combinations of sin's and cos's with that particular frequency. This is different from the eigenfunctions shown in class because in the implementation here, the operator is represented using the regular basis instead of Fourier basis, so the operator does not have the block-diagnoal form as shown in class. Therefore, the eigenfunctions are not exactly sin's and cos's but their linear combination. They are related to the fourier transform because fourier transform can be represented as linear combinations of sin's and cos's. 

The eigenvalues are sorted in descending order because it is the largest eigenvalues that have the most dominant effect. Some of the smaller ones can even be thrown away if one needs to save computation power..




## Part 2.5
The Fourier operator is block-diagnozable because it can be decomposed into different frequencies and each frequency domain has dimension 2 spanned by the $\sin$ and $\cos$ basis functions. This frequency domains correspond to the 2 by 2 blocks along the diagonal.  


## Part 3.4
This tells us that steering the function basis and apply them to the image is equivalent to steeling the image basis and apply the function to them. Essentially, this is saying Fourier transform is commutative with rotations. 