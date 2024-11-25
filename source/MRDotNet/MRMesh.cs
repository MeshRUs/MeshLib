using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using static MR.DotNet.Vector3f;

[assembly: InternalsVisibleToAttribute("MRDotNetTest")]

namespace MR.DotNet
{
    using VertCoordsReadOnly = System.Collections.ObjectModel.ReadOnlyCollection<Vector3f>;
    using VertCoords = System.Collections.Generic.List<Vector3f>;
    using TriangulationReadOnly = System.Collections.ObjectModel.ReadOnlyCollection<ThreeVertIds>;
    using Triangulation = System.Collections.Generic.List<ThreeVertIds>;
    using EdgePathReadOnly = System.Collections.ObjectModel.ReadOnlyCollection<EdgeId>;
    using EdgePath = System.Collections.Generic.List<EdgeId>;

    using VertBitSetReadOnly = BitSetReadOnly;
    using FaceBitSetReadOnly = BitSetReadOnly;

    public struct ThreeVertIds
    {
        public VertId v0;
        public VertId v1;
        public VertId v2;

        public ThreeVertIds(VertId v0_, VertId v1_, VertId v2_)
        {
            v0 = v0_;
            v1 = v1_;
            v2 = v2_;
        }

        public ThreeVertIds(int v0_, int v1_, int v2_)
        {
            v0 = new VertId( v0_ ); 
            v1 = new VertId(v1_);
            v2 = new VertId(v2_);
        }
    };

    #region C_STRUCTS

    /// parameters for \ref mrMakeTorus
    [StructLayout(LayoutKind.Sequential)]
    internal struct MRMakeTorusParameters
    {
        public float primaryRadius = 1.0f;
        public float secondaryRadius = 0.5f;
        public int primaryResolution = 32;
        public int secondaryResolution = 32;
        public MRMakeTorusParameters() { }
    };

    /// parameters for \ref mrMakeSphere
    [StructLayout(LayoutKind.Sequential)]
    internal struct MRSphereParams
    {
        public float radius = 1.0f;
        public int numMeshVertices = 100;
        public MRSphereParams() { }
    };

    [StructLayout(LayoutKind.Sequential)]
    internal struct MRTriangulation
    {
        public IntPtr data = IntPtr.Zero;
        public ulong size = 0;
        public IntPtr reserved = IntPtr.Zero;
        public MRTriangulation() { }
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct MREdgePath
    {
        public IntPtr data = IntPtr.Zero;
        public ulong size = 0;
        public IntPtr reserved = IntPtr.Zero;
        public MREdgePath() { }
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct MRPointOnFace
    {
        public FaceId face;
        public MRVector3f point;
        public MRPointOnFace() { }
    };

    [StructLayout(LayoutKind.Sequential)]
    internal struct MRTriPointf
    {
        /// barycentric coordinates:
        /// a+b in [0,1], a+b=0 => point is in v0, a+b=1 => point is on [v1,v2] edge
        /// a in [0,1], a=0 => point is on [v2,v0] edge, a=1 => point is in v1
        public float a = 0.0f;
        /// b in [0,1], b=0 => point is on [v0,v1] edge, b=1 => point is in v2
        public float b = 0.0f;
        public MRTriPointf() { }
    };

    [StructLayout(LayoutKind.Sequential)]
    internal struct MRMeshTriPoint
    {
        /// left face of this edge is considered
        public EdgeId e;
        /// barycentric coordinates
        /// \details a in [0,1], a=0 => point is on next( e ) edge, a=1 => point is in dest( e )
        /// b in [0,1], b=0 => point is on e edge, b=1 => point is in dest( next( e ) )
        /// a+b in [0,1], a+b=0 => point is in org( e ), a+b=1 => point is on prev( e.sym() ) edge
        public MRTriPointf bary;

        public MRMeshTriPoint() { }
    };

    [StructLayout(LayoutKind.Sequential)]
    internal struct MRMeshProjectionResult
    {
        /// the closest point on mesh, transformed by xf if it is given
        public MRPointOnFace proj;
        /// its barycentric representation
        public MRMeshTriPoint mtp;
        /// squared distance from pt to proj
        public float distSq = 0.0f;
        public MRMeshProjectionResult() { }
    };

    /// optional parameters for \ref mrFindProjection
    [StructLayout(LayoutKind.Sequential)]
    internal struct MRFindProjectionParameters
    {
        /// upper limit on the distance in question, if the real distance is larger than the function exits returning upDistLimitSq and no valid point
        public float upDistLimitSq = float.MaxValue;
        /// mesh-to-point transformation, if not specified then identity transformation is assumed
        public IntPtr xf = IntPtr.Zero;
        /// low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
        public float loDistLimitSq = 0.0f;

        public MRFindProjectionParameters() { }
    };
    [StructLayout(LayoutKind.Sequential)]
    internal struct MRMeshPart
    {
        public IntPtr mesh = IntPtr.Zero;
        public IntPtr region = IntPtr.Zero;
        public MRMeshPart() { }
    };


    #endregion
    /// represents a point cloud or a mesh
    public interface MeshOrPoints
    {
        public VertCoordsReadOnly Points { get; }
        public VertBitSetReadOnly ValidPoints { get; }
        public Box3f BoundingBox { get; }
    };

    /// holds together mesh/point cloud and its transformation
    public class MeshOrPointsXf
    {
        public MeshOrPoints obj;
        public AffineXf3f xf;

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrMeshOrPointsXfFromMesh(IntPtr mesh, ref MRAffineXf3f xf );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrMeshOrPointsXfFromPointCloud(IntPtr pc, ref MRAffineXf3f xf );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrMeshOrPointsXfFree(IntPtr mp);

        public MeshOrPointsXf(MeshOrPoints obj, AffineXf3f xf)
        {
            this.obj = obj;
            this.xf = xf;

            var mesh = obj as Mesh;
            if ( mesh != null) 
                mrMeshOrPointsXf_ = mrMeshOrPointsXfFromMesh(mesh.mesh_, ref xf.xf_);

            var pc = obj as PointCloud;
            if ( pc != null )
                mrMeshOrPointsXf_ = mrMeshOrPointsXfFromPointCloud(pc.pc_, ref xf.xf_);
        }

        ~MeshOrPointsXf()
        {
            mrMeshOrPointsXfFree(mrMeshOrPointsXf_);
        }

        internal IntPtr mrMeshOrPointsXf_;
    }

    public class Mesh : MeshOrPoints, IDisposable
    {
        #region C_FUNCTIONS

        /// tightly packs all arrays eliminating lone edges and invalid faces and vertices
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrMeshTopologyPack(IntPtr top);

        /// returns cached set of all valid vertices
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrMeshTopologyGetValidVerts(IntPtr top);

        /// returns cached set of all valid faces
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrMeshTopologyGetValidFaces(IntPtr top);


        /// returns three vertex ids for valid triangles (which can be accessed by FaceId),
        /// vertex ids for invalid triangles are undefined, and shall not be read
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern ref MRTriangulation mrMeshTopologyGetTriangulation(IntPtr top);

        /// returns the number of face records including invalid ones
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern ulong mrMeshTopologyFaceSize(IntPtr top);
        /// returns one edge with no valid left face for every boundary in the mesh
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern ref MREdgePath mrMeshTopologyFindHoleRepresentiveEdges(IntPtr top);

        /// gets 3 vertices of given triangular face;
        /// the vertices are returned in counter-clockwise order if look from mesh outside
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrMeshTopologyGetLeftTriVerts(IntPtr top, EdgeId a, ref VertId v0, ref VertId v1, ref VertId v2);

        /// returns the number of hole loops in the mesh;
        /// \param holeRepresentativeEdges optional output of the smallest edge id with no valid left face in every hole
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern int mrMeshTopologyFindNumHoles(IntPtr top, IntPtr holeRepresentativeEdges);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrMeshFromTriangles(IntPtr vertexCoordinates, ulong vertexCoordinatesNum, IntPtr t, ulong tNum);

        /// constructs a mesh from vertex coordinates and a set of triangles with given ids;
        /// unlike simple \ref mrMeshFromTriangles it tries to resolve non-manifold vertices by creating duplicate vertices
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrMeshFromTrianglesDuplicatingNonManifoldVertices(IntPtr vertexCoordinates, ulong vertexCoordinatesNum, IntPtr t, ulong tNum);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrMeshPoints(IntPtr mesh);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern ulong mrMeshPointsNum(IntPtr mesh);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrMakeCube(ref MRVector3f size, ref MRVector3f baseCoords);

        /// initializes a default instance
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern MRMakeTorusParameters mrMakeTorusParametersNew();

        /// creates a mesh representing a torus
        /// Z is symmetry axis of this torus
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrMakeTorus(ref MRMakeTorusParameters parameters);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrMakeTorusWithSelfIntersections(ref MRMakeTorusParameters parameters);

        /// initializes a default instance <summary>
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern MRSphereParams mrSphereParamsNew();

        /// creates a mesh of sphere with irregular triangulation
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrMakeSphere(ref MRSphereParams parameters);

        /// passes through all valid vertices and finds the minimal bounding box containing all of them;
        /// if toWorld transformation is given then returns minimal bounding box in world space
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern Box3f.MRBox3f mrMeshComputeBoundingBox(IntPtr mesh, IntPtr toWorld);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrMeshTransform(IntPtr mesh, ref MRAffineXf3f xf, IntPtr region);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern double mrMeshVolume(IntPtr mesh, IntPtr region);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrMeshPackOptimally(IntPtr mesh, bool preserveAABBTree);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrMeshTopology(IntPtr mesh);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrMeshCopy( IntPtr mesh );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrMeshFree(IntPtr mesh);       

        /// creates a default instance
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern MRFindProjectionParameters mrFindProjectionParametersNew();

        /// computes the closest point on mesh (or its region) to given point
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern MRMeshProjectionResult mrFindProjection(ref MRVector3f pt, ref MRMeshPart mp, ref MRFindProjectionParameters parameters);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrStringData(IntPtr str);

        /// gets total length of the string
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern ulong mrStringSize(IntPtr str);

        /// deallocates the string object
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrStringFree(IntPtr str);

        /// computes the area of given face-region (or whole mesh if region is null)
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern double mrMeshArea( IntPtr mesh, IntPtr region );

        /// deletes multiple given faces, also deletes adjacent edges and vertices if they were not shared by remaining faces and not in \param keepEdges
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrMeshDeleteFaces(IntPtr mesh, IntPtr fs, IntPtr keepEdges );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern float mrMeshEdgeLength( IntPtr mesh, UndirectedEdgeId e );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern float mrMeshEdgeLengthSq( IntPtr mesh, UndirectedEdgeId e );


        #endregion
        #region Constructors

        internal Mesh(IntPtr mesh)
        {
            mesh_ = mesh;
            meshTopology_ = mrMeshTopology(mesh);
        }

        internal void SkipDisposingAtFinalize()
        {
            needToDispose_ = false;
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (needToDispose_)
            {
                if (disposing)
                {
                    if ( validFaces_ is not null )
                    {
                        validFaces_.Dispose();
                        validFaces_ = null;
                    }

                    if (validPoints_ is not null)
                    {
                        validPoints_.Dispose();
                        validPoints_ = null;
                    }
                }

                if (mesh_ != IntPtr.Zero)
                {
                    mrMeshFree(mesh_);
                    mesh_ = IntPtr.Zero;
                }

                needToDispose_ = false;
            }
        }

        ~Mesh()
        {
            Dispose(false);
        }
        #endregion
        /// point coordinates
        #region Properties
        public VertCoordsReadOnly Points
        {
            get
            {
                if (points_ is null)
                {
                    int numPoints = (int)mrMeshPointsNum(mesh_);
                    points_ = new VertCoords(numPoints);
                    int sizeOfVector3f = Marshal.SizeOf(typeof(MRVector3f));

                    var pointsPtr = mrMeshPoints(mesh_);
                    for (int i = 0; i < numPoints; i++)
                    {
                        IntPtr currentPointPtr = IntPtr.Add(pointsPtr, i * sizeOfVector3f);
                        var point = Marshal.PtrToStructure<MRVector3f>(currentPointPtr);
                        points_.Add(new Vector3f(point));
                    }
                }

                return points_.AsReadOnly();
            }
        }
        /// set of all valid vertices
        public VertBitSetReadOnly ValidPoints
        {
            get
            {
                if (validPoints_ is null)
                {
                    validPoints_ = new VertBitSet(mrMeshTopologyGetValidVerts(meshTopology_));
                }
                return validPoints_;
            }
        }

        public Box3f BoundingBox
        {
            get
            {
                if (boundingBox_ is null)
                {
                    boundingBox_ = new Box3f(mrMeshComputeBoundingBox(meshTopology_, (IntPtr)null));
                }

                return boundingBox_;
            }
        }
        /// set of all valid faces
        public FaceBitSetReadOnly ValidFaces
        {
            get
            {
                if (validFaces_ is null)
                {
                    validFaces_ = new FaceBitSet(mrMeshTopologyGetValidFaces(meshTopology_));
                }
                return validFaces_;
            }
        }
        /// info about triangles
        public TriangulationReadOnly Triangulation
        {
            get
            {
                if (triangulation_ is null)
                {
                    var mrtriangulation = mrMeshTopologyGetTriangulation(meshTopology_);
                    triangulation_ = new Triangulation((int)mrtriangulation.size);
                    int sizeOfThreeVertIds = Marshal.SizeOf(typeof(ThreeVertIds));

                    var triangulationPtr = mrtriangulation.data;
                    for (int i = 0; i < triangulation_.Capacity; i++)
                    {
                        IntPtr currentTriangulationPtr = IntPtr.Add(triangulationPtr, i * sizeOfThreeVertIds);
                        triangulation_.Add(Marshal.PtrToStructure<ThreeVertIds>(currentTriangulationPtr));
                    }
                }
                return triangulation_.AsReadOnly();
            }
        }
        /// edges with no valid left face for every boundary in the mesh
        public EdgePathReadOnly HoleRepresentiveEdges
        {
            get
            {
                if (holeRepresentiveEdges_ is null)
                {
                    var mrEdges = mrMeshTopologyFindHoleRepresentiveEdges(meshTopology_);
                    holeRepresentiveEdges_ = new EdgePath((int)mrEdges.size);
                    int sizeOfEdgeId = Marshal.SizeOf(typeof(EdgeId));

                    var edgesPtr = mrEdges.data;
                    for (int i = 0; i < (int)mrEdges.size; i++)
                    {
                        IntPtr currentEdgePtr = IntPtr.Add(edgesPtr, i * sizeOfEdgeId);
                        holeRepresentiveEdges_.Add( Marshal.PtrToStructure<EdgeId>(currentEdgePtr) );
                    }
                }

                return holeRepresentiveEdges_.AsReadOnly();
            }
        }
        #endregion // Properties
        #region Methods
        /// gets 3 vertices of the left face ( face-id may not exist, but the shape must be triangular)
        /// the vertices are returned in counter-clockwise order if look from mesh outside
        public VertId[] GetLeftTriVerts(EdgeId edgeId)
        {
            VertId[] res = new VertId[3];
            VertId v0 = new VertId();
            VertId v1 = new VertId();
            VertId v2 = new VertId();

            EdgeId mrEdgeId = new EdgeId();
            mrEdgeId.Id = edgeId.Id;

            mrMeshTopologyGetLeftTriVerts(meshTopology_, mrEdgeId, ref v0, ref v1, ref v2);
            res[0].Id = v0.Id;
            res[1].Id = v1.Id;
            res[2].Id = v2.Id;

            return res;
        }
        /// transforms all points
        public void Transform(AffineXf3f xf)
        {
            mrMeshTransform(mesh_, ref xf.xf_, (IntPtr)null);
            clearManagedResources();
        }
        /// transforms all points in the region
        public void Transform(AffineXf3f xf, BitSet region)
        {
            mrMeshTransform(mesh_, ref xf.xf_, region.bs_);
            clearManagedResources();
        }
        /// packs tightly and rearranges vertices, triangles and edges to put close in space elements in close indices
        public void PackOptimally()
        {
            mrMeshPackOptimally(mesh_, true);
            clearManagedResources();
        }
        /// returns volume of whole mesh
        public double Volume()
        {
            return mrMeshVolume(mesh_, (IntPtr)null);
        }
        /// returns volume of closed mesh region, if region is not closed DBL_MAX is returned
        public double Volume(FaceBitSet region)
        {
            return mrMeshVolume(mesh_, region.bs_);
        }
        /// computes the area of given face-region (or whole mesh if region is null)
        public double Area( FaceBitSet? region = null )
        {
            return mrMeshArea( mesh_, region is null ? (IntPtr)null : region.bs_ );
        }
        /// returns Euclidean length of the edge
        public float EdgeLength(UndirectedEdgeId ue)
        {
            UndirectedEdgeId mrEdgeId = new UndirectedEdgeId();
            mrEdgeId.Id = ue.Id;
            return mrMeshEdgeLength(mesh_, mrEdgeId);
        }
        /// returns squared Euclidean length of the edge (faster to compute than length)
        public float EdgeLengthSq(UndirectedEdgeId ue)
        {
            UndirectedEdgeId mrEdgeId = new UndirectedEdgeId();
            mrEdgeId.Id = ue.Id;
            return mrMeshEdgeLengthSq(mesh_, mrEdgeId);
        }

        /// deletes multiple given faces, also deletes adjacent edges and vertices if they were not shared by remaining faces and not in \param edgesToKeep
        public void DeleteFaces( FaceBitSet faces, UndirectedEdgeBitSet? edgesToKeep = null )
        {
            mrMeshDeleteFaces(mesh_, faces.bs_, edgesToKeep is null ? (IntPtr)null : edgesToKeep.bs_);
            clearManagedResources();
        }
        /// creates a deep copy of the mesh
        public Mesh Clone()
        {
            IntPtr clonedMesh = mrMeshCopy(mesh_);
            return new Mesh(clonedMesh);
        }

        #endregion

        #region Create
        /// creates mesh from point coordinates and triangulation
        public static Mesh FromTriangles(VertCoords points, Triangulation triangles)
        {
            int sizeOfVector3f = Marshal.SizeOf(typeof(MRVector3f));
            IntPtr nativePoints = Marshal.AllocHGlobal(points.Count * sizeOfVector3f);

            int sizeOfThreeVertIds = Marshal.SizeOf(typeof(ThreeVertIds));
            IntPtr nativeTriangles = Marshal.AllocHGlobal(triangles.Count * sizeOfThreeVertIds);

            try
            {
                for (int i = 0; i < points.Count; i++)
                {
                    Marshal.StructureToPtr(points[i].vec_, IntPtr.Add(nativePoints, i * sizeOfVector3f), false);
                }

                for (int i = 0; i < triangles.Count; i++)
                {
                    Marshal.StructureToPtr(triangles[i], IntPtr.Add(nativeTriangles, i * sizeOfThreeVertIds), false);
                }

                return new Mesh(mrMeshFromTriangles(nativePoints, (ulong)points.Count, nativeTriangles, (ulong)triangles.Count));
            }
            finally
            {
                Marshal.FreeHGlobal(nativePoints);
                Marshal.FreeHGlobal(nativeTriangles);
            }
        }
        /// creates mesh from point coordinates and triangulation. If some vertices are not manifold, they will be duplicated
        public static Mesh FromTrianglesDuplicatingNonManifoldVertices(VertCoords points, Triangulation triangles)
        {
            int sizeOfVector3f = Marshal.SizeOf(typeof(MRVector3f));
            IntPtr nativePoints = Marshal.AllocHGlobal(points.Count * sizeOfVector3f);

            int sizeOfThreeVertIds = Marshal.SizeOf(typeof(ThreeVertIds));
            IntPtr nativeTriangles = Marshal.AllocHGlobal(triangles.Count * sizeOfThreeVertIds);

            try
            {
                for (int i = 0; i < points.Count; i++)
                {
                    Marshal.StructureToPtr(points[i], IntPtr.Add(nativePoints, i * sizeOfVector3f), false);
                }

                for (int i = 0; i < triangles.Count; i++)
                {
                    Marshal.StructureToPtr(triangles[i], IntPtr.Add(nativeTriangles, i * sizeOfThreeVertIds), false);
                }

                return new Mesh(mrMeshFromTrianglesDuplicatingNonManifoldVertices(nativePoints, (ulong)points.Count, nativeTriangles, (ulong)triangles.Count));
            }
            finally
            {
                Marshal.FreeHGlobal(nativePoints);
                Marshal.FreeHGlobal(nativeTriangles);
            }
        }        
       
       
        /// creates a parallelepiped with given sizes and base
        public static Mesh MakeCube(Vector3f size, Vector3f baseCoords)
        {
            return new Mesh(mrMakeCube(ref size.vec_, ref baseCoords.vec_));
        }
        /// creates a sphere of given radius and vertex count
        public static Mesh MakeSphere(float radius, int vertexCount)
        {
            MRSphereParams mrSphereParams = new MRSphereParams();
            mrSphereParams.radius = radius;
            mrSphereParams.numMeshVertices = vertexCount;
            return new Mesh(mrMakeSphere(ref mrSphereParams));
        }
        /// creates a torus with given parameters
        public static Mesh MakeTorus(float primaryRadius, float secondaryRadius, int primaryResolution, int secondaryResolution)
        {
            MRMakeTorusParameters mrMakeTorusParameters = new MRMakeTorusParameters();
            mrMakeTorusParameters.primaryRadius = primaryRadius;
            mrMakeTorusParameters.secondaryRadius = secondaryRadius;
            mrMakeTorusParameters.primaryResolution = primaryResolution;
            mrMakeTorusParameters.secondaryResolution = secondaryResolution;
            return new Mesh(mrMakeTorus(ref mrMakeTorusParameters));
        }

        /// creates a torus with self-intersections
        internal static Mesh MakeTorusWithSelfIntersections(float primaryRadius, float secondaryRadius, int primaryResolution, int secondaryResolution)
        {
            MRMakeTorusParameters mrMakeTorusParameters = new MRMakeTorusParameters();
            mrMakeTorusParameters.primaryRadius = primaryRadius;
            mrMakeTorusParameters.secondaryRadius = secondaryRadius;
            mrMakeTorusParameters.primaryResolution = primaryResolution;
            mrMakeTorusParameters.secondaryResolution = secondaryResolution;
            return new Mesh(mrMakeTorusWithSelfIntersections(ref mrMakeTorusParameters));
        }

        #endregion
        #region Find projection
        /// computes the closest point on mesh (or its region) to given point
        static public MeshProjectionResult FindProjection(Vector3f point, MeshPart meshPart, float maxDistanceSquared = float.MaxValue, AffineXf3f? xf = null, float minDistanceSquared = 0.0f)
        {
            MRFindProjectionParameters mrParams = new MRFindProjectionParameters();
            mrParams.loDistLimitSq = minDistanceSquared;
            mrParams.upDistLimitSq = maxDistanceSquared;
            mrParams.xf = xf is not null ? xf.XfAddr() : (IntPtr)null;

            var mrRes = mrFindProjection(ref point.vec_, ref meshPart.mrMeshPart, ref mrParams);

            MeshProjectionResult result = new MeshProjectionResult();
            result.distanceSquared = mrRes.distSq;

            result.pointOnFace = new PointOnFace();
            result.pointOnFace.point = new Vector3f(mrRes.proj.point);
            result.pointOnFace.faceId.Id = mrRes.proj.face.Id;

            result.meshTriPoint = new MeshTriPoint();
            result.meshTriPoint.e.Id = mrRes.mtp.e.Id;
            result.meshTriPoint.bary.a = mrRes.mtp.bary.a;
            result.meshTriPoint.bary.b = mrRes.mtp.bary.b;

            return result;
        }
        #endregion


        #region Private fields

        void clearManagedResources()
        {            
            if (validFaces_ is not null)
            {
                validFaces_.Dispose();
                validFaces_ = null;
            }

            if (validPoints_ is not null)
            {
                validPoints_.Dispose();
                validPoints_ = null;
            }

            points_ = null;
            triangulation_ = null;
            holeRepresentiveEdges_ = null;
            boundingBox_ = null;
        }

        internal IntPtr varMesh()
        {
            clearManagedResources();
            return mesh_;
        }

        internal IntPtr mesh_;
        internal IntPtr meshTopology_;
        private bool needToDispose_ = true;

        private VertCoords? points_;
        private VertBitSet? validPoints_;
        private FaceBitSet? validFaces_;
        private Triangulation? triangulation_;
        private EdgePath? holeRepresentiveEdges_;
        private Box3f? boundingBox_;
        #endregion

    }
    public struct TriPoint
    {
        /// a in [0,1], a=0 => point is on [v2,v0] edge, a=1 => point is in v1
        public float a;
        /// b in [0,1], b=0 => point is on [v0,v1] edge, b=1 => point is in v2
        public float b;
        /// given three values in three vertices, computes interpolated value at this barycentric coordinates
        public Vector3f Interpolate(Vector3f p0, Vector3f p1, Vector3f p2)
        {
            return p0 * (1 - a - b) + a * p1 + b * p2;
        }
    };

    public struct MeshTriPoint
    {
        public EdgeId e;
        public TriPoint bary;
    }

    public struct PointOnFace
    {
        public FaceId faceId;
        public Vector3f point;
    };

    public struct MeshProjectionResult
    {
        public PointOnFace pointOnFace;
        public MeshTriPoint meshTriPoint;
        public float distanceSquared;
    };

    public class MeshPart
    {
        public Mesh mesh;
        public FaceBitSet? region;

        internal MRMeshPart mrMeshPart;
        public MeshPart(Mesh mesh, FaceBitSet? region = null)
        {
            this.mesh = mesh;
            this.region = region;

            mrMeshPart.mesh = mesh.mesh_;
            mrMeshPart.region = region is null ? (IntPtr)null : region.bs_;
        }
    };
}
