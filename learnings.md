Some learnings from my ascii neural renderer, finished as of this diff

Had 3 layers: one-hot (dimension 128), hidden layer (dimension 64), then output is bitmap 1 & 0 (dimension 48, 8x6 grid). Zeroed a char’s one-hot during training but still perfect. Realized the bias from the hidden layer perfected retained all the info (`Wx + b` where `x` is all 0). Removed that bias, still perfect, because of bias on the output layer. Removed that and finally output for that char’s funky looking. I like this self-repairing thing from a neural net. Most self-healing architectures in traditional programming are either low level (tcp retries) or super complex (distributed system) or tedious (data migration/skew)

Made two chars have the same one-hot encoding. Funky output ofc

Tried with a nonexistent one-hot vector (random char that isn’t part of the training data). Outputs roughly a B because it’s what’s shared between most ascii chars

Learned about some PyTorch internals and revisited some much needed linear algebra. If I don’t need torch’s flexibility I think it’s pretty easy to beat it with a compute shader

Tinkered with the hidden layer size til I realized dim 5 works but dim 4 doesn’t. Then realized I train on 27 distinct chars (a-Z and space) and `2^4<27<2^5` aka 5 bits needed to represent each char. But one-hot encoding represents a char. So I removed the entire hidden layer. Soon I’ll change the one hot encoding to this 5 bits representation, which is literally just the binary integer representation of a char

Tried some visualization tools

**More importantly**:

Tried turning one-hot encoding of my 27 ascii chars into a binary integer encoding, aka assign integer to each char. My instinct is to cut down on unnecessary state space (dimensions) for the input layer by using how every computer encodes integers; seems obvious enough. However, results degraded _hard_. LLMs keep telling me that it's because I've accidentally forced in the assumption that some chars' integers are closer to others: say A is 000, B is 001, D is 011, well B and D are "closer" (1 bit difference) than A and D (2 bit difference). I get that but struggle to understand _how_ this affects the model. The "closer together than others" explanation doesn't give me the right mental model. But I finally discovered 3 better explanations:
1. If you view these 3 bits not as binary representation of an integer, but as a 3-dimensional vectors on a cube, then the length of the vectors aren't the same (`||(0,0,1)|| < ||(0,1,1)||`). Which means during matrix multiplication, for each dot product, some "integers" (aka vectors) get higher dot products unfairly. The weights struggle to undo that length bias you've built in.
2. For `Wx` where `x` is one-hot encoded, the matmul result is equivalent to `x` selecting a column of `W`. Aka each column's weights only need to satisfy that particular one-hot 1 bit (that particular char). They don't have to compromise and satisfy other chars, unlike for binary integer encoding where e.g. 011 selects _two_ columns and those weights need to fight. Here's a simplified calculation: say my input can be either A or B and output is 1 bit, true (1) for A and false (0) for B. With one-hot, A=01 B=10, weight matrix is `[0 1]`. If I use binary integer, A=0 B=1, weight matrix is just a single cell `[0.5]` that's never correct (ignore bias and activation). Aka if the problem isn't easily compressible, we literally don't have enough bits to calculate the solution.

Say we wanna blindly rectify our original integer-encoded 27 chars network by adding a hidden layer. As an upper bound, said layer would need 27 dimensions, to essentially undo the biased integer encoding back into one-hot. Lower bound... should be `log2(27) = 4.75 ~= 5` neurons? To represent `2^5 = 32 > 27` patterns? I've experimentally verified this lower bound calculation by brute forcing a small network.

Now I'm wondering if our traditional BPE tokenization is also biased and took various layers just to unbias it.
