using NUnit.Framework;

namespace MR.DotNet.Test
{
    [TestFixture]
    internal class ExpandShrinkTests
    {
        [Test]
        public void TestExpandShrink()
        {
            var mesh = Mesh.MakeSphere(1.0f, 3000);
            var region = ExpandShrink.Expand(mesh, new FaceId(0), 3);
            Assert.That(region.Count, Is.EqualTo(75));
            ExpandShrink.Expand(mesh, region, 3);
            Assert.That(region.Count, Is.GreaterThan(75));
            ExpandShrink.Shrink(mesh, region, 3);
            Assert.That(region.Count, Is.EqualTo(75));
        }

        [Test]
        public void TestExpandShrinkVerts()
        {
            var mesh = Mesh.MakeSphere(1.0f, 3000);
            var region = ExpandShrink.Expand(mesh, new VertId(0), 3);
            Assert.That(region.Count, Is.EqualTo(37));
            ExpandShrink.Expand(mesh, region, 3);
            Assert.That(region.Count, Is.GreaterThan(37));
            ExpandShrink.Shrink(mesh, region, 3);
            Assert.That(region.Count, Is.EqualTo(37));
        }

    }
}
