using NUnit.Framework;

namespace MR.DotNet.Test
{
    [TestFixture]
    internal class BitSetTests
    {
        [Test]
        public void TestDefaultConstructor()
        {
            var a = new MR.DotNet.BitSet();
            Assert.That( a.Size(), Is.EqualTo(0) );
        }

        [Test]
        public void TestConstructor()
        {
            var a = new MR.DotNet.BitSet( 20 );
            Assert.That(a.Size(), Is.EqualTo(20));
        }

        [Test]
        public void TestSet()
        {
            var a = new MR.DotNet.BitSet( 10 );
            a.Set( 5 );
            Assert.That( a.Test( 5 ) );
            Assert.That( !a.Test( 4 ) );
        }

        [Test]
        public void TestFindLast()
        {
            var a = new MR.DotNet.BitSet( 10 );
            a.Set( 5 );
            Assert.That( a.FindLast(), Is.EqualTo(5) );
        }

        [Test]
        public void TestAutoResize()
        {
            var a = new MR.DotNet.BitSet();
            a.AutoResizeSet(6);
            Assert.That( a.Size() == 7 );
        }

        [Test]
        public void TestSubtraction()
        {
            var a = new MR.DotNet.BitSet( 10 );
            a.Set( 5 );
            a.Set(6);
            var b = new MR.DotNet.BitSet( 10 );
            b.Set( 6 );
            var c = a - b;
            Assert.That( c.Test( 5 ) );
            Assert.That( !c.Test( 6 ) );
        }

        [Test]
        public void TestUnion()
        {
            var a = new MR.DotNet.BitSet( 10 );
            a.Set( 5 );            
            var b = new MR.DotNet.BitSet( 10 );
            b.Set( 6 );
            var c = a | b;
            Assert.That( c.Test( 5 ) );
            Assert.That( c.Test( 6 ) );
        }
    }
}
