using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace MR.DotNet
{
    /// container of bits with read-only access
    public abstract class BitSetReadOnly
    {
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern ulong mrBitSetSize( IntPtr bs );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern ulong mrBitSetCount( IntPtr bs );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        [return: MarshalAs(UnmanagedType.I1)]
        private static extern bool mrBitSetEq( IntPtr a, IntPtr b );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        [return: MarshalAs(UnmanagedType.I1)]
        private static extern bool mrBitSetTest( IntPtr bs, ulong index );       

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern ulong mrBitSetFindFirst( IntPtr bs );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern ulong mrBitSetFindLast( IntPtr bs );

        internal IntPtr bs_;
        /// test if given bit is set
        public bool Test( int i )
        {
            var result = mrBitSetTest( bs_, (ulong)i );
            return result;
        }
        /// returns index of the first set bit
        public int FindFirst()
        {
            return (int)mrBitSetFindFirst( bs_ );
        }
        /// returns index of the last set bit
        public int FindLast()
        {
            return (int)mrBitSetFindLast( bs_ );
        }
        /// returns total number of bits
        public int Size()
        {
            return (int)mrBitSetSize( bs_ );
        }
        /// returns number of set bits
        public int Count()
        {
            return (int)mrBitSetCount( bs_ );
        }
        /// returns a deep copy of the bitset
        public abstract BitSetReadOnly Clone();
        /// checks if two bitsets are equal (have the same length and identical bit values)
        public static bool operator ==(BitSetReadOnly a, BitSetReadOnly b )
        {
            return mrBitSetEq( a.bs_, b.bs_ );
        }
        /// checks if two bitsets are not equal
        public static bool operator !=(BitSetReadOnly a, BitSetReadOnly b )
        {
            return !mrBitSetEq( a.bs_, b.bs_ );
        }

        public override bool Equals( object obj )
        {
            return (obj is BitSetReadOnly) ? this == (BitSetReadOnly)obj : false;
        }

        public override int GetHashCode()
        {
            throw new NotImplementedException();
        }
    }
    /// container of bits with full access
    public class BitSet : BitSetReadOnly, IDisposable
    {
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrBitSetCopy(IntPtr bs);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrBitSetNew(ulong numBits, bool fillValue);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern void mrBitSetSet(IntPtr bs, ulong index, bool value);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern void mrBitSetResize(IntPtr bs, ulong size, bool value);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern void mrBitSetAutoResizeSet(IntPtr bs, ulong pos, bool value);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrBitSetSub(IntPtr a, IntPtr b);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrBitSetOr(IntPtr a, IntPtr b);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern void mrBitSetFree(IntPtr bs);

        /// creates empty bitset
        public BitSet() : this( 0, false )
        {
        }

        internal BitSet(IntPtr bs)
        {
            bs_ = bs;
        }
        /// creates bitset with given size
        public BitSet( int size ) : this( size, false )
        {
        }
        /// creates bitset with given size and fill value
        public BitSet( int size, bool fillValue )
        {
            bs_ = mrBitSetNew((ulong)size, fillValue);
            needDispose = true;
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (needDispose)
            {
                if (bs_ != IntPtr.Zero)
                {
                    Console.WriteLine("mrBitSetFree start");
                    mrBitSetFree(bs_);
                    Console.WriteLine("mrBitSetFree end");
                    bs_ = IntPtr.Zero;
                }

                needDispose = false;
            }
        }

        ~BitSet()
        {
            Dispose(false);
        }
        /// sets the given bit to true
        public void Set(int index)
        {
            mrBitSetSet(bs_, (ulong)index, true);
        }
        /// sets the given bit to value
        public void Set( int index, bool value )
        {
            mrBitSetSet( bs_, (ulong)index, value );
        }
        /// changes the size of the bitset
        public void Resize( int size )
        {
            mrBitSetResize( bs_, (ulong)size, false );
        }
        /// changes the size of the bitset, sets new bits to value
        public void Resize( int size, bool value )
        {
            mrBitSetResize( bs_, (ulong)size, value );
        }
        /// sets element pos to given value, adjusting the size of the set to include new element if necessary
        public void AutoResizeSet( int pos )
        {
            mrBitSetAutoResizeSet( bs_, (ulong)pos, true );
        }
        /// sets element pos to given value, adjusting the size of the set to include new element if necessary, sets new bits to value
        public void AutoResizeSet( int pos, bool value )
        {
            mrBitSetAutoResizeSet( bs_, (ulong)pos, value );
        }
        /// returns a deep copy of the bitset
        public override BitSetReadOnly Clone()
        {
            IntPtr bsCopy = mrBitSetCopy( bs_ );
            return new BitSet(bsCopy);
        }
        /// creates a new bitset including a's bits and excluding b's bits
        public static BitSet operator -(BitSet a, BitSet b)
        {
            return new BitSet(mrBitSetSub(a.bs_, b.bs_));
        }

        /// creates a new bitset including both a's bits and  b's bits
        public static BitSet operator |(BitSet a, BitSet b)
        {
            return new BitSet(mrBitSetOr(a.bs_, b.bs_));
        }

        bool needDispose = false;
    }
    /// container of bits representing vert indices
    public class VertBitSet : BitSet 
    {
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrBitSetCopy(IntPtr bs);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrBitSetSub(IntPtr a, IntPtr b);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrBitSetOr(IntPtr a, IntPtr b);

        internal VertBitSet(IntPtr bs) : base(bs) { }
        public VertBitSet( int size ) : base( size ) { }
        public VertBitSet( int size, bool fillValue ) : base( size, fillValue ) { }

        /// returns a deep copy of the bitset
        public override BitSetReadOnly Clone()
        {
            IntPtr bsCopy = mrBitSetCopy(bs_);
            return new VertBitSet(bsCopy);
        }
        /// creates a new bitset including a's bits and excluding b's bits 
        public static VertBitSet operator -(VertBitSet a, VertBitSet b)
        {
            return new VertBitSet(mrBitSetSub(a.bs_, b.bs_));
        }

        /// creates a new bitset including both a's bits and  b's bits
        public static VertBitSet operator |(VertBitSet a, VertBitSet b)
        {
            return new VertBitSet(mrBitSetOr(a.bs_, b.bs_));
        }
    }
    /// container of bits representing face indices
    public class FaceBitSet : BitSet
    {
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrBitSetCopy(IntPtr bs);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrBitSetSub(IntPtr a, IntPtr b);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrBitSetOr(IntPtr a, IntPtr b);

        internal FaceBitSet(IntPtr bs) : base(bs) { }
        /// creates bitset with given size
        public FaceBitSet( int size ) : base( size ) { }
        /// creates bitset with given size and fill value
        public FaceBitSet( int size, bool fillValue ) : base( size, fillValue ) { }
        /// returns a deep copy of the bitset
        public override BitSetReadOnly Clone()
        {
            IntPtr bsCopy = mrBitSetCopy(bs_);
            return new FaceBitSet(bsCopy);
        }
        /// creates a new bitset including a's bits and excluding b's bits 
        public static FaceBitSet operator -(FaceBitSet a, FaceBitSet b)
        {
            return new FaceBitSet(mrBitSetSub(a.bs_, b.bs_));
        }

        /// creates a new bitset including both a's bits and  b's bits
        public static FaceBitSet operator |(FaceBitSet a, FaceBitSet b)
        {
            return new FaceBitSet(mrBitSetOr(a.bs_, b.bs_));
        }
    }
    /// container of bits representing edge indices
    public class EdgeBitSet : BitSet
    {
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrBitSetCopy(IntPtr bs);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrBitSetSub(IntPtr a, IntPtr b);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrBitSetOr(IntPtr a, IntPtr b);

        internal EdgeBitSet(IntPtr bs) : base(bs) { }
        /// creates bitset with given size
        public EdgeBitSet( int size ) : base( size ) { }
        /// creates bitset with given size and fill value
        public EdgeBitSet( int size, bool fillValue ) : base( size, fillValue ) { }
        /// returns a deep copy of the bitset
        public override BitSetReadOnly Clone()
        {
            IntPtr bsCopy = mrBitSetCopy(bs_);
            return new EdgeBitSet(bsCopy);
        }
        /// creates a new bitset including a's bits and excluding b's bits 
        public static EdgeBitSet operator -(EdgeBitSet a, EdgeBitSet b)
        {
            return new EdgeBitSet(mrBitSetSub(a.bs_, b.bs_));
        }

        /// creates a new bitset including both a's bits and  b's bits
        public static EdgeBitSet operator |(EdgeBitSet a, EdgeBitSet b)
        {
            return new EdgeBitSet(mrBitSetOr(a.bs_, b.bs_));
        }

    }
    /// container of bits representing undirected edge indices
    public class UndirectedEdgeBitSet : BitSet
    {
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrBitSetCopy(IntPtr bs);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrBitSetSub(IntPtr a, IntPtr b);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrBitSetOr(IntPtr a, IntPtr b);

        internal UndirectedEdgeBitSet(IntPtr bs) : base(bs) { }
        /// creates bitset with given size
        public UndirectedEdgeBitSet( int size ) : base( size ) { }
        /// creates bitset with given size and fill value
        public UndirectedEdgeBitSet( int size, bool fillValue ) : base( size, fillValue ) { }
        /// returns a deep copy of the bitset
        public override BitSetReadOnly Clone()
        {
            IntPtr bsCopy = mrBitSetCopy(bs_);
            return new UndirectedEdgeBitSet(bsCopy);
        }
        /// creates a new bitset including a's bits and excluding b's bits 
        public static UndirectedEdgeBitSet operator -(UndirectedEdgeBitSet a, UndirectedEdgeBitSet b)
        {
            return new UndirectedEdgeBitSet(mrBitSetSub(a.bs_, b.bs_));
        }

        /// creates a new bitset including both a's bits and  b's bits
        public static UndirectedEdgeBitSet operator |(UndirectedEdgeBitSet a, UndirectedEdgeBitSet b)
        {
            return new UndirectedEdgeBitSet(mrBitSetOr(a.bs_, b.bs_));
        }
    }
}
