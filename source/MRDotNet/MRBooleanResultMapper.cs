using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Runtime.InteropServices;

namespace MR.DotNet
{
    using VertMap = List<VertId>;
    using VertMapReadOnly = ReadOnlyCollection<VertId>;

    using FaceMap = List<FaceId>;
    using FaceMapReadOnly = ReadOnlyCollection<FaceId>;

    public enum MapObject
    {
        A = 0,
        B = 1,
        Count = 2
    };

    [StructLayout(LayoutKind.Sequential)]
    internal struct MRFaceMap
    {
        public IntPtr data = IntPtr.Zero;
        public ulong size = 0;
        public IntPtr reserved = IntPtr.Zero;
        public MRFaceMap(){}
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct MRVertMap
    {
        public IntPtr data = IntPtr.Zero;
        public ulong size = 0;
        public IntPtr reserved = IntPtr.Zero;
        public MRVertMap() { }
    };

    public class BooleanMaps
    {
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern MRFaceMap mrBooleanResultMapperMapsCut2origin(IntPtr maps);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern MRFaceMap mrBooleanResultMapperMapsCut2newFaces(IntPtr maps);

        
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern MRVertMap mrBooleanResultMapperMapsOld2NewVerts(IntPtr maps);

        /// old topology indexes are valid if true
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern bool mrBooleanResultMapperMapsIdentity(IntPtr maps);

        internal BooleanMaps(IntPtr maps)
        {
            maps_ = maps;
        }
        /// "after cut" faces to "origin" faces
        /// this map is not 1-1, but N-1.
        /// This map can contain invalid faces (-1). Please check validity before use.
        public FaceMapReadOnly Cut2Origin
        {
            get
            {
                if ( cut2origin_ is null )
                {
                    var mrMap = mrBooleanResultMapperMapsCut2origin(maps_);
                    cut2origin_ = new List<FaceId>( (int)mrMap.size );
                    int sizeOfFaceId= Marshal.SizeOf(typeof(FaceId));
                    for ( int i = 0; i < (int)mrMap.size; i++ )
                    {
                        IntPtr currentFacePtr = IntPtr.Add(mrMap.data, i * sizeOfFaceId);
                        var face = Marshal.PtrToStructure<FaceId>(currentFacePtr);
                        cut2origin_.Add(new FaceId(face.Id));
                    }
                }

                return cut2origin_.AsReadOnly();
            }
        }

        /// "after cut" faces to "after stitch" faces (1-1). This map can contain invalid faces (-1). Please check validity before use.
        public FaceMapReadOnly Cut2NewFaces
        {
            get
            {
                if ( cut2newFaces_ is null )
                {
                    var mrMap = mrBooleanResultMapperMapsCut2newFaces(maps_);
                    cut2newFaces_ = new List<FaceId>( (int)mrMap.size );
                    int sizeOfFaceId= Marshal.SizeOf(typeof(FaceId));
                    for ( int i = 0; i < (int)mrMap.size; i++ )
                    {
                        IntPtr currentFacePtr = IntPtr.Add(mrMap.data, i * sizeOfFaceId);
                        var face = Marshal.PtrToStructure<FaceId>(currentFacePtr);
                        cut2newFaces_.Add(new FaceId(face.Id));
                    }
                }

                return cut2newFaces_.AsReadOnly();
            }
        }

        /// "origin" vertices to "after stitch" vertices (1-1). This map can contain invalid vertices (-1). Please check validity before use.
        public VertMapReadOnly Old2NewVerts
        {
            get
            {
                if ( old2newVerts_ is null )
                {
                    var mrMap = mrBooleanResultMapperMapsOld2NewVerts(maps_);
                    old2newVerts_ = new List<VertId>( (int)mrMap.size );
                    int sizeOfVertId= Marshal.SizeOf(typeof(VertId));
                    for ( int i = 0; i < (int)mrMap.size; i++ )
                    {
                        IntPtr currentVertPtr = IntPtr.Add(mrMap.data, i * sizeOfVertId);
                        var vert = Marshal.PtrToStructure<VertId>(currentVertPtr);
                        old2newVerts_.Add(new VertId(vert.Id));
                    }
                }

                return old2newVerts_.AsReadOnly();
            }
        }
        /// old topology indices are valid if true
        public bool Identity
        {
            get
            {
                return mrBooleanResultMapperMapsIdentity(maps_);
            }
        }

        IntPtr maps_;
        FaceMap? cut2origin_;
        FaceMap? cut2newFaces_;
        VertMap? old2newVerts_;
    }
    ///this class allows to map faces, vertices and edges of mesh `A` and mesh `B` input of MeshBoolean to result mesh topology primitives
    public class BooleanResultMapper : IDisposable
    {
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrBooleanResultMapperNew();

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrBooleanResultMapperMapFaces(IntPtr mapper, IntPtr oldBS, MapObject obj);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrBooleanResultMapperMapVerts(IntPtr mapper, IntPtr oldBS, MapObject obj);


        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrBooleanResultMapperNewFaces(IntPtr mapper);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrBooleanResultMapperFilteredOldFaceBitSet(IntPtr mapper, IntPtr oldBS, MapObject obj);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrBooleanResultMapperGetMaps(IntPtr mapper, MapObject index);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrBooleanResultMapperFree(IntPtr mapper);

        #region constructor and destructor


        public BooleanResultMapper()
        {
            mapper_ = mrBooleanResultMapperNew();
        }

        private bool disposed = false;
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposed)
            {
                if (mapper_ != IntPtr.Zero)
                {
                    mrBooleanResultMapperFree(mapper_);
                }

                disposed = true;
            }
        }

        ~BooleanResultMapper()
        {
            Dispose(false);
        }
        #endregion
        #region properties

        /// returns faces bitset of result mesh corresponding input one
        public FaceBitSet FaceMap( FaceBitSet oldBS, MapObject obj)
        {
            if ( maps_ is null )
                maps_ = new BooleanMaps?[2];

            if ( maps_[(int)obj] is null )
            {
                maps_[(int)obj] = new BooleanMaps( mrBooleanResultMapperGetMaps(mapper_, obj) );
            }

            return new FaceBitSet(mrBooleanResultMapperMapFaces(mapper_, oldBS.bs_, obj));
        }
        /// Returns vertices bitset of result mesh corresponding input one
        public VertBitSet VertMap( VertBitSet oldBS, MapObject obj)
        {
            if (maps_ is null)
                maps_ = new BooleanMaps?[2];

            if (maps_[(int)obj] is null)
            {
                maps_[(int)obj] = new BooleanMaps(mrBooleanResultMapperGetMaps(mapper_, obj));
            }

            return new VertBitSet(mrBooleanResultMapperMapVerts(mapper_, oldBS.bs_, obj));
        }
        /// Returns only new faces that are created during boolean operation
        public FaceBitSet NewFaces()
        {
            return new FaceBitSet(mrBooleanResultMapperNewFaces(mapper_));
        }

        public BooleanMaps GetMaps( MapObject obj)
        {
            if (maps_ is null)
                maps_ = new BooleanMaps?[2];

            var res = maps_[(int)obj];
            if (res is null)
            {
                res = maps_[(int)obj] = new BooleanMaps(mrBooleanResultMapperGetMaps(mapper_, obj));
            }

            return res;
        }
        /// returns updated oldBS leaving only faces that has corresponding ones in result mesh
        public FaceBitSet FilteredOldFaceBitSet( FaceBitSet oldBS, MapObject obj)
        {
            if (maps_ is null)
                maps_ = new BooleanMaps?[2];

            if (maps_[(int)obj] is null)
            {
                maps_[(int)obj] = new BooleanMaps(mrBooleanResultMapperGetMaps(mapper_, obj));
            }

            return new FaceBitSet(mrBooleanResultMapperFilteredOldFaceBitSet(mapper_, oldBS.bs_, obj));
        }

        internal IntPtr Mapper { get { return mapper_; } }
        #endregion
        #region private fields

        IntPtr mapper_;
        BooleanMaps?[]? maps_;
        #endregion
    }
}
