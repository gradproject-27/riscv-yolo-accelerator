// 1 + 1 on the Hummingbird E203 core.
// volatile stops the compiler from working out the answer at compile time,
// so the core really executes the add instruction.

volatile int result;

int main(void)
{
    volatile int a = 1;
    volatile int b = 1;
    result = a + b;
    return result;
}
